"""
Generate baseline FlashInfer solutions for all definitions missing one.
Writes solutions to tmp/flashinfer-trace/solutions/baseline/{op_type}/{def_name}/{name}.json
"""

import hashlib
import json
import os
import re
import sys

BASELINE_DIR = "tmp/flashinfer-trace/solutions/baseline"
DEF_DIR = "flashinfer_trace/definitions"

HARDWARE = ["NVIDIA A100", "NVIDIA H20", "NVIDIA H100", "NVIDIA H200", "NVIDIA B200"]


def sol_name(def_name: str) -> str:
    h = hashlib.sha256(def_name.encode()).hexdigest()[:6]
    return f"flashinfer_wrapper_{h}"


def write_solution(op_type: str, def_name: str, content: str, description: str):
    name = sol_name(def_name)
    sol = {
        "name": name,
        "definition": def_name,
        "author": "flashinfer",
        "spec": {
            "language": "python",
            "target_hardware": HARDWARE,
            "entry_point": "main.py::run",
            "dependencies": ["flashinfer"],
            "destination_passing_style": False,
        },
        "sources": [{"path": "main.py", "content": content}],
        "description": description,
    }
    out_dir = os.path.join(BASELINE_DIR, op_type, def_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(sol, f, indent=4)
    print(f"  wrote {out_path}")


# ── rmsnorm ──────────────────────────────────────────────────────────────────


def gen_rmsnorm(def_name: str):
    m = re.match(r"rmsnorm_h(\d+)", def_name)
    hidden = int(m.group(1))
    content = f"""\
import torch
import flashinfer


def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape

    assert hidden_size == {hidden}

    EPS = 1e-6

    output = flashinfer.norm.rmsnorm(hidden_states, weight, eps=EPS)

    return output
"""
    write_solution("rmsnorm", def_name, content, f"FlashInfer rmsnorm baseline for {def_name}.")


def gen_fused_add_rmsnorm(def_name: str):
    m = re.match(r"fused_add_rmsnorm_h(\d+)", def_name)
    hidden = int(m.group(1))
    content = f"""\
import torch
import flashinfer


def run(hidden_states, residual, weight):
    batch_size, hidden_size = hidden_states.shape

    assert hidden_size == {hidden}

    EPS = 1e-5

    # fused_add_rmsnorm modifies hidden_states and residual in-place
    flashinfer.norm.fused_add_rmsnorm(hidden_states, residual, weight, EPS)

    return hidden_states
"""
    write_solution(
        "rmsnorm", def_name, content, f"FlashInfer fused_add_rmsnorm baseline for {def_name}."
    )


# ── gemm ─────────────────────────────────────────────────────────────────────


def gen_gemm(def_name: str):
    content = """\
import torch
import torch.nn.functional as F


def run(A: torch.Tensor, B: torch.Tensor):
    return F.linear(A, B)
"""
    write_solution(
        "gemm", def_name, content, f"torch.nn.functional.linear baseline for {def_name}."
    )


# ── gqa_paged decode ─────────────────────────────────────────────────────────

DECODE_TEMPLATE = """\
import torch
import flashinfer

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {}
_wrapper_cache = {}
_plan_state = {}


def _get_workspace(device):
    key = str(device)
    buffer = _workspace_cache.get(key)
    if buffer is None or buffer.device != device or buffer.numel() < _WORKSPACE_SIZE_BYTES:
        buffer = torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buffer
    return buffer


def _get_wrapper(key, device):
    wrapper = _wrapper_cache.get(key)
    if wrapper is None:
        workspace = _get_workspace(device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
        )
        _wrapper_cache[key] = wrapper
    return wrapper


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    device = q.device
    wrapper_key = (
        str(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q.dtype,
        k_cache.dtype,
    )

    wrapper = _get_wrapper(wrapper_key, device)
    state = _plan_state.get(wrapper_key)

    needs_plan = True
    if state is not None:
        needs_plan = (
            state.get("batch_size") != batch_size
            or state.get("len_indptr") != len_indptr
            or state.get("num_kv_indices") != num_kv_indices
            or state.get("sm_scale") != sm_scale
            or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
            or state.get("kv_indices_ptr") != kv_indices.data_ptr()
        )

    if needs_plan:
        kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
        wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode="NONE",
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            sm_scale=sm_scale,
        )
        _plan_state[wrapper_key] = {
            "batch_size": batch_size,
            "len_indptr": len_indptr,
            "num_kv_indices": num_kv_indices,
            "sm_scale": sm_scale,
            "kv_indptr_ptr": kv_indptr.data_ptr(),
            "kv_indices_ptr": kv_indices.data_ptr(),
        }

    output, lse = wrapper.run(
        q,
        (k_cache, v_cache),
        return_lse=True,
    )

    return output, lse
"""


DECODE_EXPAND_TEMPLATE = """\
import torch
import flashinfer

# group_size={group_size} ({num_q} qo_heads / {num_kv} kv_heads) is not natively supported.
# Work-around: expand KV heads from {num_kv} to {num_q} (repeat_interleave x{group_size})
# so the wrapper sees group_size=1 (MHA), which is mathematically equivalent.

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {{}}
_wrapper_cache = {{}}
_plan_state = {{}}


def _get_workspace(device):
    key = str(device)
    buffer = _workspace_cache.get(key)
    if buffer is None or buffer.device != device or buffer.numel() < _WORKSPACE_SIZE_BYTES:
        buffer = torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buffer
    return buffer


def _get_wrapper(key, device):
    wrapper = _wrapper_cache.get(key)
    if wrapper is None:
        workspace = _get_workspace(device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
        _wrapper_cache[key] = wrapper
    return wrapper


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    group_size = num_qo_heads // num_kv_heads
    # Expand KV heads: [pages, page_size, {num_kv}, head_dim] -> [pages, page_size, {num_q}, head_dim]
    k_exp = k_cache.repeat_interleave(group_size, dim=2)
    v_exp = v_cache.repeat_interleave(group_size, dim=2)
    expanded_kv_heads = num_qo_heads  # {num_q}

    device = q.device
    wkey = (str(device), num_qo_heads, expanded_kv_heads, head_dim, page_size, q.dtype, k_exp.dtype)
    wrapper = _get_wrapper(wkey, device)
    state = _plan_state.get(wkey)

    needs_plan = True
    if state is not None:
        needs_plan = (
            state.get("batch_size") != batch_size
            or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
            or state.get("kv_indices_ptr") != kv_indices.data_ptr()
            or state.get("sm_scale") != sm_scale
        )

    if needs_plan:
        kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
        wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=expanded_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode="NONE",
            q_data_type=q.dtype,
            kv_data_type=k_exp.dtype,
            sm_scale=sm_scale,
        )
        _plan_state[wkey] = {{
            "batch_size": batch_size,
            "kv_indptr_ptr": kv_indptr.data_ptr(),
            "kv_indices_ptr": kv_indices.data_ptr(),
            "sm_scale": sm_scale,
        }}

    output, lse = wrapper.run(q, (k_exp, v_exp), return_lse=True)
    return output, lse
"""

# Supported native group_sizes for BatchDecodeWithPagedKVCacheWrapper
_DECODE_NATIVE_GROUP_SIZES = {1, 2, 4, 8}


def gen_gqa_paged_decode(def_name: str):
    m = re.match(r"gqa_paged_decode_h(\d+)_kv(\d+)_d(\d+)", def_name)
    num_q, num_kv = int(m.group(1)), int(m.group(2))
    group_size = num_q // num_kv
    if group_size in _DECODE_NATIVE_GROUP_SIZES:
        write_solution(
            "gqa_paged",
            def_name,
            DECODE_TEMPLATE,
            f"FlashInfer BatchDecodeWithPagedKVCacheWrapper baseline for {def_name}.",
        )
    else:
        content = DECODE_EXPAND_TEMPLATE.format(group_size=group_size, num_q=num_q, num_kv=num_kv)
        write_solution(
            "gqa_paged",
            def_name,
            content,
            f"FlashInfer BatchDecodeWithPagedKVCacheWrapper baseline for {def_name}. "
            f"KV heads expanded x{group_size} for unsupported group_size.",
        )


# ── gqa_paged prefill ────────────────────────────────────────────────────────

PREFILL_TEMPLATE = """\
import torch
import flashinfer

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {}
_wrapper_cache = {}
_plan_state = {}


def _get_workspace(device):
    key = str(device)
    buffer = _workspace_cache.get(key)
    if buffer is None or buffer.device != device or buffer.numel() < _WORKSPACE_SIZE_BYTES:
        buffer = torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buffer
    return buffer


def _get_wrapper(key, device):
    wrapper = _wrapper_cache.get(key)
    if wrapper is None:
        workspace = _get_workspace(device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
        )
        _wrapper_cache[key] = wrapper
    return wrapper


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    batch_size = qo_indptr.shape[0] - 1
    num_kv_indices = kv_indices.shape[0]

    device = q.device
    wrapper_key = (
        str(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q.dtype,
        k_cache.dtype,
    )

    wrapper = _get_wrapper(wrapper_key, device)
    state = _plan_state.get(wrapper_key)

    if isinstance(sm_scale, torch.Tensor):
        sm_scale_value = float(sm_scale.item())
    else:
        sm_scale_value = float(sm_scale)

    needs_plan = True
    if state is not None:
        needs_plan = (
            state.get("total_q") != total_q
            or state.get("batch_size") != batch_size
            or state.get("num_kv_indices") != num_kv_indices
            or state.get("sm_scale") != sm_scale_value
            or state.get("qo_indptr_ptr") != qo_indptr.data_ptr()
            or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
            or state.get("kv_indices_ptr") != kv_indices.data_ptr()
        )

    if needs_plan:
        last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=True,
            sm_scale=sm_scale,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
        )
        _plan_state[wrapper_key] = {
            "total_q": total_q,
            "batch_size": batch_size,
            "num_kv_indices": num_kv_indices,
            "sm_scale": sm_scale_value,
            "qo_indptr_ptr": qo_indptr.data_ptr(),
            "kv_indptr_ptr": kv_indptr.data_ptr(),
            "kv_indices_ptr": kv_indices.data_ptr(),
        }

    output, lse = wrapper.run(
        q,
        (k_cache, v_cache),
        return_lse=True,
    )

    return output, lse
"""


PREFILL_TEMPLATE_WITH_LAST_PAGE_LEN = """\
import torch
import flashinfer

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {}
_wrapper_cache = {}
_plan_state = {}


def _get_workspace(device):
    key = str(device)
    buffer = _workspace_cache.get(key)
    if buffer is None or buffer.device != device or buffer.numel() < _WORKSPACE_SIZE_BYTES:
        buffer = torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buffer
    return buffer


def _get_wrapper(key, device):
    wrapper = _wrapper_cache.get(key)
    if wrapper is None:
        workspace = _get_workspace(device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
        )
        _wrapper_cache[key] = wrapper
    return wrapper


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    batch_size = kv_indptr.shape[0] - 1
    num_kv_indices = kv_indices.shape[0]

    device = q.device
    wrapper_key = (
        str(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q.dtype,
        k_cache.dtype,
    )

    wrapper = _get_wrapper(wrapper_key, device)
    state = _plan_state.get(wrapper_key)

    if isinstance(sm_scale, torch.Tensor):
        sm_scale_value = float(sm_scale.item())
    else:
        sm_scale_value = float(sm_scale)

    needs_plan = True
    if state is not None:
        needs_plan = (
            state.get("total_q") != total_q
            or state.get("batch_size") != batch_size
            or state.get("num_kv_indices") != num_kv_indices
            or state.get("sm_scale") != sm_scale_value
            or state.get("qo_indptr_ptr") != qo_indptr.data_ptr()
            or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
            or state.get("kv_indices_ptr") != kv_indices.data_ptr()
        )

    if needs_plan:
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=True,
            sm_scale=sm_scale,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
        )
        _plan_state[wrapper_key] = {
            "total_q": total_q,
            "batch_size": batch_size,
            "num_kv_indices": num_kv_indices,
            "sm_scale": sm_scale_value,
            "qo_indptr_ptr": qo_indptr.data_ptr(),
            "kv_indptr_ptr": kv_indptr.data_ptr(),
            "kv_indices_ptr": kv_indices.data_ptr(),
        }

    output, lse = wrapper.run(
        q,
        (k_cache, v_cache),
        return_lse=True,
    )

    return output, lse
"""


def gen_gqa_paged_prefill(def_name: str):
    # ps>1 definitions include kv_last_page_len as an explicit input
    m = re.match(r"gqa_paged_prefill_causal_.*_ps(\d+)$", def_name)
    page_size = int(m.group(1)) if m else 1
    if page_size > 1:
        template = PREFILL_TEMPLATE_WITH_LAST_PAGE_LEN
    else:
        template = PREFILL_TEMPLATE
    write_solution(
        "gqa_paged",
        def_name,
        template,
        f"FlashInfer BatchPrefillWithPagedKVCacheWrapper baseline for {def_name}.",
    )


# ── gqa_ragged prefill ───────────────────────────────────────────────────────


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


RAGGED_STANDARD = """\
import torch
import flashinfer

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {}
_wrapper_cache = {}
_plan_state = {}


def _get_workspace(device):
    key = str(device)
    buffer = _workspace_cache.get(key)
    if buffer is None or buffer.device != device or buffer.numel() < _WORKSPACE_SIZE_BYTES:
        buffer = torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buffer
    return buffer


def _get_wrapper(key, device):
    wrapper = _wrapper_cache.get(key)
    if wrapper is None:
        workspace = _get_workspace(device)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
        )
        _wrapper_cache[key] = wrapper
    return wrapper


def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    batch_size = qo_indptr.shape[0] - 1

    device = q.device
    wrapper_key = (
        str(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q.dtype,
        k.dtype,
        v.dtype,
    )

    wrapper = _get_wrapper(wrapper_key, device)
    state = _plan_state.get(wrapper_key)

    needs_plan = True
    if state is not None:
        needs_plan = (
            state.get("total_q") != total_q
            or state.get("total_kv") != total_kv
            or state.get("batch_size") != batch_size
            or state.get("sm_scale") != sm_scale
            or state.get("qo_indptr_ptr") != qo_indptr.data_ptr()
            or state.get("kv_indptr_ptr") != kv_indptr.data_ptr()
        )

    if needs_plan:
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=True,
            sm_scale=sm_scale,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )
        _plan_state[wrapper_key] = {
            "total_q": total_q,
            "total_kv": total_kv,
            "batch_size": batch_size,
            "sm_scale": sm_scale,
            "qo_indptr_ptr": qo_indptr.data_ptr(),
            "kv_indptr_ptr": kv_indptr.data_ptr(),
        }

    output, lse = wrapper.run(q, k, v, return_lse=True)
    return output, lse
"""

RAGGED_EXPAND_TEMPLATE = """\
import torch
import flashinfer

# group_size={group_size} ({num_q} qo_heads / {num_kv} kv_heads) is not a power-of-2.
# Work-around: expand KV heads from {num_kv} to {num_q} (repeat_interleave x{group_size})
# so group_size=1 (MHA), which is mathematically equivalent.

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {{}}
_wrapper_cache = {{}}
_plan_state = {{}}


def _get_workspace(device):
    key = str(device)
    buf = _workspace_cache.get(key)
    if buf is None:
        buf = torch.empty(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buf
    return buf


def _get_wrapper(key, device):
    w = _wrapper_cache.get(key)
    if w is None:
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_get_workspace(device), kv_layout="NHD")
        _wrapper_cache[key] = w
    return w


def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    device = q.device
    group_size = num_qo_heads // num_kv_heads  # {group_size}
    # Expand KV heads: [total_kv, {num_kv}, {head_dim}] -> [total_kv, {num_q}, {head_dim}]
    k_exp = k.repeat_interleave(group_size, dim=1)
    v_exp = v.repeat_interleave(group_size, dim=1)
    expanded_heads = num_qo_heads  # {num_q}
    wkey = (str(device), num_qo_heads, expanded_heads, head_dim, q.dtype, k.dtype)
    wrapper = _get_wrapper(wkey, device)
    state = _plan_state.get(wkey)
    needs_plan = state is None or state["total_q"] != total_q or state["qo_ptr"] != qo_indptr.data_ptr()
    if needs_plan:
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=expanded_heads,
            head_dim_qk=head_dim,
            causal=True,
            sm_scale=float(sm_scale),
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )
        _plan_state[wkey] = {{"total_q": total_q, "qo_ptr": qo_indptr.data_ptr()}}
    output, lse = wrapper.run(q, k_exp, v_exp, return_lse=True)
    return output, lse
"""


def gen_gqa_ragged(def_name: str):
    # parse h{Q}_kv{KV}_d{D}
    m = re.match(r"gqa_ragged_prefill_causal_h(\d+)_kv(\d+)_d(\d+)", def_name)
    num_q, num_kv, head_dim = int(m.group(1)), int(m.group(2)), int(m.group(3))
    group_size = num_q // num_kv
    if _is_pow2(group_size):
        content = RAGGED_STANDARD
        desc = f"FlashInfer BatchPrefillWithRaggedKVCacheWrapper baseline for {def_name}."
    else:
        content = RAGGED_EXPAND_TEMPLATE.format(
            group_size=group_size, num_q=num_q, num_kv=num_kv, head_dim=head_dim
        )
        desc = (
            f"FlashInfer BatchPrefillWithRaggedKVCacheWrapper baseline for {def_name}. "
            f"KV heads expanded x{group_size} for non-power-of-2 group_size."
        )
    write_solution("gqa_ragged", def_name, content, desc)


# ── sampling ──────────────────────────────────────────────────────────────────

TOP_K_TEMPLATE = """\
import torch
import flashinfer


def run(probs, top_k):
    batch_size, vocab_size = probs.shape
    device = probs.device

    assert vocab_size == {vocab}

    probs = probs.to(torch.float32)

    samples = flashinfer.sampling.top_k_sampling_from_probs(
        probs=probs,
        top_k=top_k,
        indices=None,
        deterministic=False,
        generator=None,
        check_nan=False
    )

    samples = samples.to(torch.int64)

    return samples
"""

TOP_P_TEMPLATE = """\
import torch
import flashinfer


def run(probs, top_p):
    batch_size, vocab_size = probs.shape
    device = probs.device

    assert vocab_size == {vocab}

    probs = probs.to(torch.float32)

    samples = flashinfer.sampling.top_p_sampling_from_probs(
        probs=probs,
        top_p=top_p,
        indices=None,
        deterministic=False,
        generator=None,
        check_nan=False
    )

    samples = samples.to(torch.int64)

    return samples
"""

TOP_K_TOP_P_TEMPLATE = """\
import torch
import flashinfer


def run(probs, top_k, top_p):
    batch_size, vocab_size = probs.shape
    device = probs.device

    assert vocab_size == {vocab}

    probs = probs.to(torch.float32)

    samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs=probs,
        top_k=top_k,
        top_p=top_p,
        indices=None,
        deterministic=False,
        generator=None,
        check_nan=False
    )

    samples = samples.to(torch.int64)

    return samples
"""


def gen_sampling(def_name: str):
    m = re.match(r"(top_k_top_p|top_k|top_p)_sampling_from_probs_v(\d+)", def_name)
    kind, vocab = m.group(1), int(m.group(2))
    if kind == "top_k":
        content = TOP_K_TEMPLATE.format(vocab=vocab)
        desc = f"FlashInfer top_k_sampling_from_probs baseline for {def_name}."
    elif kind == "top_p":
        content = TOP_P_TEMPLATE.format(vocab=vocab)
        desc = f"FlashInfer top_p_sampling_from_probs baseline for {def_name}."
    else:
        content = TOP_K_TOP_P_TEMPLATE.format(vocab=vocab)
        desc = f"FlashInfer top_k_top_p_sampling_from_probs baseline for {def_name}."
    write_solution("sampling", def_name, content, desc)


# ── main ──────────────────────────────────────────────────────────────────────


def get_missing(op_type: str):
    defs = {
        f.replace(".json", "")
        for f in os.listdir(os.path.join(DEF_DIR, op_type))
        if f.endswith(".json")
    }
    baselines_dir = os.path.join(BASELINE_DIR, op_type)
    existing = set(os.listdir(baselines_dir)) if os.path.isdir(baselines_dir) else set()
    return sorted(defs - existing)


def main():
    total = 0

    print("=== rmsnorm ===")
    for def_name in get_missing("rmsnorm"):
        if def_name.startswith("fused_add_rmsnorm"):
            gen_fused_add_rmsnorm(def_name)
        else:
            gen_rmsnorm(def_name)
        total += 1

    print("=== gemm ===")
    for def_name in get_missing("gemm"):
        gen_gemm(def_name)
        total += 1

    print("=== gqa_paged ===")
    for def_name in get_missing("gqa_paged"):
        if "_decode_" in def_name:
            gen_gqa_paged_decode(def_name)
        else:
            gen_gqa_paged_prefill(def_name)
        total += 1

    print("=== gqa_ragged ===")
    for def_name in get_missing("gqa_ragged"):
        gen_gqa_ragged(def_name)
        total += 1

    print("=== sampling ===")
    for def_name in get_missing("sampling"):
        gen_sampling(def_name)
        total += 1

    print(f"\nDone: generated {total} baseline solutions.")


if __name__ == "__main__":
    main()
