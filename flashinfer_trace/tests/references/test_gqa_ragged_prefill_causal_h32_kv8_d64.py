"""Reference test for gqa_ragged_prefill_causal_h32_kv8_d64 (Llama 3.2 1B)."""

import math
from pathlib import Path

import flashinfer
import torch

from flashinfer_bench.data import Definition, load_json_file

DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64


def load_definition(name: str) -> Definition:
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)
    raise FileNotFoundError(f"Definition {name} not found in {DEFINITIONS_DIR}")


def compile_reference(reference_code: str):
    namespace = {"torch": torch, "math": math}
    exec(reference_code, namespace)
    return namespace["run"]


def generate_random_inputs(batch_size, max_q_len, max_kv_len, device="cuda"):
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    total_q = int(qo_indptr[-1].item())
    total_kv = int(kv_indptr[-1].item())

    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "sm_scale": sm_scale,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, atol=1e-2, rtol=5e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return False

    definition = load_definition("gqa_ragged_prefill_causal_h32_kv8_d64")
    run = compile_reference(definition.reference)
    inputs = generate_random_inputs(batch_size, max_q_len, max_kv_len, device)

    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["sm_scale"],
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )
    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=True,
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], inputs["k"], inputs["v"], return_lse=True)

    out_ok = torch.allclose(ref_o.float(), fi_output.float(), atol=atol, rtol=rtol)
    lse_ok = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    return out_ok and lse_ok


def main():
    configs = [(1, 16, 32), (4, 32, 64), (8, 64, 128)]
    passed = sum(1 for b, q, k in configs if test_correctness(b, q, k))
    print(f"{passed}/{len(configs)} passed")


if __name__ == "__main__":
    main()
