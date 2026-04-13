#!/usr/bin/env python3
"""
Sanitize FlashInfer tensor dumps into workload JSONL format.

Reads a FlashInfer Level 10 dump directory and converts matching API call
dumps into workload JSONL entries following the flashinfer-trace schema.

Tensor storage policy (for SGLang-collected dumps):
  - int32/int64 tensors (structural: indptrs, indices) → saved as safetensors
  - float tensors (activations: q, k, v, etc.)        → {"type": "random"}
  - null-shape inputs (scalars: sm_scale, eps, etc.)  → {"type": "scalar", "value": ...}

For wrapper class APIs (e.g., BatchPrefillWithPagedKVCacheWrapper), plan() dumps
are paired with run() dumps (same PID, plan immediately precedes run) to extract
structural int32/int64 tensors (qo_indptr, paged_kv_indptr, paged_kv_indices,
paged_kv_last_page_len).

Usage:
    python sanitize_dumps.py \\
        --dump-dir ./workload_dumps_20260326_123456 \\
        --definitions gqa_paged_prefill_causal_h20_kv4_d128_ps64 \\
        --flashinfer-trace-dir ~/flashinfer-trace \\
        --replace

    # Skip const-axis check when collecting TP=1 dumps for a TP=2 definition:
    python sanitize_dumps.py \\
        --dump-dir ./workload_dumps_tp1 \\
        --definitions gqa_paged_prefill_causal_h20_kv4_d128_ps64 \\
        --flashinfer-trace-dir ~/flashinfer-trace \\
        --replace \\
        --skip-const-axis-check
"""

import argparse
import json
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import torch

# Global flag: skip const-axis shape verification (set via --skip-const-axis-check).
# Useful when collecting from TP=1 SGLang to populate a TP=2 definition where
# head-count const axes (num_qo_heads, num_kv_heads) differ but structural
# tensors (indptrs, indices) are identical.
_SKIP_CONST_AXIS_CHECK: bool = False

# Plan() kwarg names → definition input names
_PLAN_KWARG_TO_DEF: dict[str, str] = {
    "qo_indptr": "qo_indptr",
    "paged_kv_indptr": "kv_indptr",
    "paged_kv_indices": "kv_indices",
    "paged_kv_last_page_len": "kv_last_page_len",
    # Ragged wrapper uses these directly
    "kv_indptr": "kv_indptr",
    "kv_indices": "kv_indices",
    # BatchDecodeWithPagedKVCacheWrapper.plan uses shorter names
    "indptr": "kv_indptr",
    "indices": "kv_indices",
    "last_page_len": "kv_last_page_len",
}

_INT_DTYPES = {"torch.int32", "torch.int64", "int32", "int64"}
_FLOAT_DTYPES = {
    "torch.float32",
    "torch.float16",
    "torch.bfloat16",
    "torch.float64",
    "float32",
    "float16",
    "bfloat16",
    "float64",
}


def _parse_param_names(sig_str: str) -> list[str]:
    """Parse parameter names in order from a function signature string."""
    # Strip surrounding parens and return type annotation
    sig_str = sig_str.strip()
    if sig_str.startswith("("):
        sig_str = sig_str[1:]
    sig_str = re.sub(r"\)\s*->.*$", "", sig_str).rstrip(")")

    params = []
    depth = 0
    current = ""
    for ch in sig_str + ",":
        if ch in "([{":
            depth += 1
            current += ch
        elif ch in ")]}":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            token = current.strip()
            if token:
                # Extract param name before ':', '=', or whitespace
                name = re.split(r"[:\s=]", token)[0].strip()
                if (
                    name
                    and name != "*"
                    and name != "/"
                    and not name.startswith("**")
                    and not name.startswith("*")
                ):
                    params.append(name)
            current = ""
        else:
            current += ch
    return params


def _get_tensor_from_dump(
    tensors: dict,
    metadata: dict,
    param_names: list[str],
    param_name: str,
    def_input_index: int | None = None,
) -> torch.Tensor | None:
    """
    Get a tensor by its parameter name from a dump's tensor dict.
    Handles both kwarg_ and arg_ style keys.

    Falls back to positional lookup using def_input_index when the definition
    input name (e.g. 'hidden_states') doesn't match the function signature
    parameter name (e.g. 'input').
    """
    # Try kwarg first (preferred — unambiguous)
    kwarg_key = f"kwarg_{param_name}"
    if kwarg_key in tensors:
        return tensors[kwarg_key]

    # Try positional arg using param index from function signature
    if param_name in param_names:
        idx = param_names.index(param_name)
        arg_key = f"arg_{idx}"
        if arg_key in tensors:
            return tensors[arg_key]

    # Fallback: use definition input position when name doesn't match signature
    # e.g. definition has 'hidden_states' but signature has 'input' at same position
    if def_input_index is not None:
        arg_key = f"arg_{def_input_index}"
        if arg_key in tensors:
            return tensors[arg_key]

    return None


def _get_scalar_from_dump(
    metadata: dict, param_names: list[str], param_name: str, def_input_index: int | None = None
):
    """Get a non-tensor scalar value from dump metadata."""
    input_meta = metadata.get("input_metadata", {})

    kwarg_key = f"kwarg_{param_name}"
    if kwarg_key in input_meta:
        return input_meta[kwarg_key]

    if param_name in param_names:
        idx = param_names.index(param_name)
        arg_key = f"arg_{idx}"
        if arg_key in input_meta:
            return input_meta[arg_key]

    # Fallback: use definition input position
    if def_input_index is not None:
        arg_key = f"arg_{def_input_index}"
        if arg_key in input_meta:
            return input_meta[arg_key]

    return None


def _load_input_tensors(call_dir: Path) -> dict[str, torch.Tensor]:
    """Load input tensors from a call dump directory."""
    safetensors_path = call_dir / "inputs.safetensors"
    pt_path = call_dir / "inputs.pt"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu")
    elif pt_path.exists():
        return torch.load(str(pt_path), map_location="cpu")
    else:
        return {}


def _infer_axis_value(
    tensors: dict,
    metadata: dict,
    param_names: list[str],
    axis_name: str,
    axis_def: dict,
    definition: dict,
) -> int | None:
    """
    Infer the value of a variable axis from tensor shapes.

    Scans all definition inputs whose shapes reference this axis and
    extracts the corresponding dimension value from the actual tensors.
    """
    # Collect candidate values from all inputs whose shape references this axis
    candidates = []
    for def_idx, (input_name, input_spec) in enumerate(definition.get("inputs", {}).items()):
        shape_template = input_spec.get("shape")
        if shape_template is None:
            continue
        for dim_idx, dim_name in enumerate(shape_template):
            if dim_name == axis_name:
                tensor = _get_tensor_from_dump(
                    tensors, metadata, param_names, input_name, def_input_index=def_idx
                )
                if tensor is not None and dim_idx < len(tensor.shape):
                    candidates.append(tensor.shape[dim_idx])

    if not candidates:
        return None

    # All candidates should agree
    if len(set(candidates)) > 1:
        print(
            f"  WARNING: axis '{axis_name}' has inconsistent values across tensors: {set(candidates)}"
        )
    return candidates[0]


def _verify_constant_axis(
    tensors: dict,
    metadata: dict,
    param_names: list[str],
    axis_name: str,
    axis_def: dict,
    definition: dict,
    skip_const_axis_check: bool = False,
) -> bool:
    """Verify that a constant axis matches all observed tensor shapes."""
    if skip_const_axis_check:
        return True

    expected_val = axis_def.get("value")
    if expected_val is None:
        return True  # Can't verify without expected value

    for def_idx, (input_name, input_spec) in enumerate(definition.get("inputs", {}).items()):
        shape_template = input_spec.get("shape")
        if shape_template is None:
            continue
        for dim_idx, dim_name in enumerate(shape_template):
            if dim_name == axis_name:
                tensor = _get_tensor_from_dump(
                    tensors, metadata, param_names, input_name, def_input_index=def_idx
                )
                if tensor is not None and dim_idx < len(tensor.shape):
                    actual = tensor.shape[dim_idx]
                    if actual != expected_val:
                        print(
                            f"  WARNING: constant axis '{axis_name}' expected {expected_val}, "
                            f"got {actual} (from input '{input_name}')"
                        )
                        return False
    return True


def process_call_dump(
    call_dir: Path,
    definition: dict,
    def_name: str,
    blob_base_dir: Path,
    func_name_in_dump: str,
    plan_dir: Path | None = None,
) -> dict | None:
    """
    Process a single call dump directory and produce a workload JSONL entry.

    Returns None if the call doesn't match the expected definition or is invalid.

    For BatchPrefillWithPagedKVCacheWrapper.run() dumps, pass the corresponding
    plan_dir to supplement missing structural tensors (qo_indptr, kv_indptr,
    kv_indices, kv_last_page_len) from the plan() call.
    """
    metadata_path = call_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return None

    # Read last record (most complete state)
    last_record = None
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    last_record = json.loads(line)
                except json.JSONDecodeError:
                    continue

    if last_record is None:
        return None

    # Check function name matches
    recorded_func_name = last_record.get("function_name", "")
    if recorded_func_name != func_name_in_dump:
        return None

    # Only process completed dumps (both inputs and outputs saved)
    if last_record.get("execution_status") not in ("completed", "inputs_saved"):
        return None

    # Load input tensors
    tensors = _load_input_tensors(call_dir)
    if not tensors:
        print(f"  WARNING: no input tensors found in {call_dir}")
        return None

    # For wrapper class run() dumps: supplement with structural tensors from paired plan() dump.
    # plan() captures int32/int64 structural tensors (qo_indptr, paged_kv_indptr,
    # paged_kv_indices, paged_kv_last_page_len); run() only has the query tensor.
    plan_record = None
    if plan_dir is not None and func_name_in_dump.endswith(".run"):
        plan_tensors = _load_input_tensors(plan_dir)
        plan_meta_path = plan_dir / "metadata.jsonl"
        if plan_meta_path.exists() and plan_tensors:
            with open(plan_meta_path) as pf:
                for line in pf:
                    line = line.strip()
                    if line:
                        try:
                            plan_record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
            if plan_record:
                # plan() signature: (self, qo_indptr, paged_kv_indptr, paged_kv_indices,
                #                    paged_kv_last_page_len, num_qo_heads, ...)
                # These are logged as arg_1..arg_4 (arg_0 = self)
                plan_sig = plan_record.get("function_signature", "")
                plan_param_names = _parse_param_names(plan_sig) if plan_sig else []
                # Extract plan tensors using _PLAN_KWARG_TO_DEF reverse mapping:
                # def input name → plan param name(s) that produce it
                plan_injected = {}
                for plan_param, def_key in _PLAN_KWARG_TO_DEF.items():
                    if def_key not in plan_injected:
                        t = _get_tensor_from_dump(
                            plan_tensors, plan_record, plan_param_names, plan_param
                        )
                        if t is not None and str(t.dtype) in _INT_DTYPES:
                            plan_injected[def_key] = t

                # kv_indices from SGLang's pool is over-allocated: shape is [pool_size] but
                # only the first kv_indptr[-1] entries are valid. Trim to valid range.
                if "kv_indices" in plan_injected and "kv_indptr" in plan_injected:
                    valid_count = int(plan_injected["kv_indptr"][-1].item())
                    plan_injected["kv_indices"] = plan_injected["kv_indices"][:valid_count].clone()

                # kv_last_page_len from plan() has shape [batch_size], but some definitions
                # store it as shape [len_indptr] = [batch_size + 1]. Pad if needed.
                if "kv_last_page_len" in plan_injected and "qo_indptr" in plan_injected:
                    expected_len = plan_injected["qo_indptr"].shape[0]  # = len_indptr
                    t = plan_injected["kv_last_page_len"]
                    if t.shape[0] < expected_len:
                        pad = torch.zeros(expected_len - t.shape[0], dtype=t.dtype, device=t.device)
                        plan_injected["kv_last_page_len"] = torch.cat([t, pad])

                for def_key, t in plan_injected.items():
                    # Inject into tensors dict using kwarg_ prefix so _get_tensor_from_dump
                    # finds them via the kwarg_<name> lookup path.
                    tensors[f"kwarg_{def_key}"] = t
                # Also inject sm_scale from plan kwarg if available
                if "kwarg_sm_scale" not in tensors:
                    sm_val = plan_record.get("input_metadata", {}).get("kwarg_sm_scale")
                    if sm_val is not None:
                        last_record.setdefault("input_metadata", {})["kwarg_sm_scale"] = sm_val

    # Get parameter names from function signature
    sig_str = last_record.get("function_signature", "")
    param_names = _parse_param_names(sig_str) if sig_str else []

    # Extract axes
    axes = {}
    valid = True
    # For paged prefill run() dumps, skip const axis check: k_cache and v_cache shapes
    # are never captured (they live in the full KV pool, not direct args), so the fallback
    # index lookup would pick up the wrong tensor (q) and give false negatives.
    skip_check = _SKIP_CONST_AXIS_CHECK or (plan_dir is not None)
    for axis_name, axis_def in definition.get("axes", {}).items():
        if axis_def["type"] == "var":
            val = _infer_axis_value(
                tensors, last_record, param_names, axis_name, axis_def, definition
            )
            if val is None:
                print(f"  WARNING: could not infer variable axis '{axis_name}'")
            else:
                axes[axis_name] = int(val)
        elif axis_def["type"] == "const":
            if not _verify_constant_axis(
                tensors,
                last_record,
                param_names,
                axis_name,
                axis_def,
                definition,
                skip_const_axis_check=skip_check,
            ):
                valid = False
                break

    if not valid:
        return None

    # For paged prefill: num_pages can't be reliably inferred from k_cache (not captured).
    # The k_cache fallback wrongly uses q.shape[0] = total_q. Override with max(kv_indices)+1
    # — the minimum pool size that can hold all referenced page IDs.
    if plan_dir is not None:
        kv_indices_t = tensors.get("kwarg_kv_indices")
        if kv_indices_t is not None and len(kv_indices_t) > 0:
            axes["num_pages"] = int(kv_indices_t.max().item()) + 1

    # Generate UUID for this workload
    workload_uuid = str(uuid.uuid4())
    safetensors_filename = f"{def_name}_{workload_uuid}.safetensors"

    # Determine blob output path
    op_type_for_blob = definition.get("op_type", "unknown")
    blob_dir = blob_base_dir / op_type_for_blob / def_name
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / safetensors_filename

    # Collect all inputs for the workload
    op_type = definition.get("op_type", "unknown")
    relative_path = f"./blob/workloads/{op_type}/{def_name}/{safetensors_filename}"

    workload_inputs = {}
    tensors_to_save = {}

    for def_idx, (input_name, input_spec) in enumerate(definition.get("inputs", {}).items()):
        is_optional = input_spec.get("optional", False)
        shape_template = input_spec.get("shape")

        if shape_template is None:
            # Scalar input (shape is null)
            scalar_val = _get_scalar_from_dump(
                last_record, param_names, input_name, def_input_index=def_idx
            )
            if scalar_val is not None:
                # Unwrap if it was serialized as a dict
                if isinstance(scalar_val, dict):
                    scalar_val = scalar_val.get("value", scalar_val)
                workload_inputs[input_name] = {
                    "type": "scalar",
                    "value": float(scalar_val) if scalar_val is not None else 1.0 / (128**0.5),
                }
            else:
                # Fallback: use default scale 1/sqrt(head_size)
                workload_inputs[input_name] = {"type": "scalar", "value": 0.08838834764831843}
            continue

        # Determine tensor storage based on dtype (definition spec):
        #   float → "random" (activation values don't affect kernel performance)
        #   int32/int64 → save as safetensors (structural: indices, indptrs)
        input_dtype = input_spec.get("dtype", "float32")
        if input_dtype in {"bfloat16", "float16", "float32", "float64"}:
            workload_inputs[input_name] = {"type": "random"}
            continue

        # int32/int64: look up tensor (prefer kwarg_ prefix, then positional arg)
        # For wrapper class run() dumps, structural tensors were injected from plan().
        # When plan_dir is set, skip the def_input_index fallback to avoid wrongly
        # matching float activations (q) at the same positional index.
        tensor = _get_tensor_from_dump(tensors, last_record, param_names, input_name)

        if tensor is None and plan_dir is None:
            # Plain function path: allow def_input_index fallback
            tensor = _get_tensor_from_dump(
                tensors, last_record, param_names, input_name, def_input_index=def_idx
            )

        if tensor is None:
            if is_optional:
                continue
            else:
                print(f"  WARNING: int tensor '{input_name}' not found in dump {call_dir.name}")
                workload_inputs[input_name] = {"type": "random"}
                continue

        # Save int tensor to safetensors blob
        tensors_to_save[input_name] = tensor.contiguous()
        workload_inputs[input_name] = {
            "type": "safetensors",
            "path": relative_path,
            "tensor_key": input_name,
        }

    if not tensors_to_save and not workload_inputs:
        print(f"  WARNING: no inputs for {def_name}")
        return None
    if not tensors_to_save:
        # All inputs are random/scalar — still valid, no blob file needed
        # Remove the blob path entries from workload_inputs since no file will be written
        return {
            "definition": def_name,
            "solution": None,
            "workload": {"uuid": workload_uuid, "axes": axes, "inputs": workload_inputs},
            "evaluation": None,
        }

    # Save all tensors to a single safetensors file
    # Clone to ensure no shared storage (safetensors rejects shared-memory tensors)
    from safetensors.torch import save_file

    tensors_to_save = {k: v.contiguous().clone() for k, v in tensors_to_save.items()}
    save_file(tensors_to_save, str(blob_path))

    return {
        "definition": def_name,
        "solution": None,
        "workload": {"uuid": workload_uuid, "axes": axes, "inputs": workload_inputs},
        "evaluation": None,
    }


def _cleanup_orphaned_blobs(jsonl_path: Path, trace_dir: Path, def_name: str, op_type: str) -> None:
    """Delete blob safetensors files no longer referenced by the existing JSONL."""
    blob_dir = trace_dir / "blob" / "workloads" / op_type / def_name
    if not blob_dir.exists():
        return

    # Collect UUIDs referenced in the existing JSONL
    referenced_uuids = set()
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        referenced_uuids.add(entry["workload"]["uuid"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    except OSError:
        return

    # Delete orphaned blob files
    deleted = 0
    for blob_file in blob_dir.glob("*.safetensors"):
        if not any(u in blob_file.name for u in referenced_uuids):
            blob_file.unlink()
            deleted += 1

    if deleted:
        print(f"  Cleaned up {deleted} orphaned blob files from {blob_dir}")


def sanitize_dumps(
    dump_dir: Path, definition_files: list[Path], flashinfer_trace_dir: Path, replace: bool = False
) -> dict[str, list[dict]]:
    """
    Process all call dumps in dump_dir for matching definitions.

    Returns a dict mapping def_name → list of workload JSONL entries.
    """
    # Load all definitions
    definitions = {}
    func_name_to_defs = defaultdict(list)  # func_name → [def_name, ...]

    for def_file in definition_files:
        with open(def_file) as f:
            defn = json.load(f)

        def_name = defn["name"]
        definitions[def_name] = defn

        # Build function name → definition mapping using fi_api tags.
        # FlashInfer may log wrapper class methods as either "ClassName.run" or just "run";
        # index both so the lookup works regardless.
        for tag in defn.get("tags", []):
            if tag.startswith("fi_api:"):
                api_path = tag[len("fi_api:") :]
                last = api_path.split(".")[-1]
                if last[0].isupper():
                    if "Ragged" in last:
                        # RaggedKVCacheWrapper: in recent FlashInfer, @flashinfer_api is on
                        # run(), so logs show ".run". forward()/forward_return_lse() are
                        # deprecated wrappers that call run() — kept for old builds.
                        func_name_to_defs[f"{last}.run"].append(def_name)
                        func_name_to_defs[f"{last}.forward"].append(def_name)
                        func_name_to_defs[f"{last}.forward_return_lse"].append(def_name)
                        func_name_to_defs["run"].append(def_name)
                        func_name_to_defs["forward"].append(def_name)
                        func_name_to_defs["forward_return_lse"].append(def_name)
                    else:
                        # Qualified name (e.g. "BatchPrefillWithPagedKVCacheWrapper.run")
                        func_name_to_defs[f"{last}.run"].append(def_name)
                        # Unqualified fallback (e.g. "run") — in case FlashInfer logs just the method name
                        func_name_to_defs["run"].append(def_name)
                else:
                    func_name_to_defs[last].append(def_name)

    if not func_name_to_defs:
        print("WARNING: No fi_api tags found in any definition. Matching by shape only.")

    # Blob base directory (op_type appended per-definition in process_call_dump)
    blob_base_dir = flashinfer_trace_dir / "blob" / "workloads"

    # Find all call directories in dump_dir
    call_dirs = sorted(
        [d for d in dump_dir.iterdir() if d.is_dir() and (d / "metadata.jsonl").exists()],
        key=lambda d: d.name,
    )

    print(f"Found {len(call_dirs)} call dumps in {dump_dir}")

    # Build a time-ordered plan() index for wrapper class pairing.
    # plan() is called once per forward batch; run() is called once per layer.
    # We pair each run() call with the most recent plan() call for the same pid.
    # Structural tensors (qo_indptr, kv_indptr, kv_indices) from plan() are identical
    # for all run() calls in the same forward pass.
    # Key: pid → list of (dir_name, dir_path) sorted by dir_name (timestamp prefix)
    plan_dirs_by_pid: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for d in call_dirs:
        m = re.search(r"_pid(\d+)_(.+?)_call(\d+)$", d.name)
        if m and m.group(2).endswith(".plan"):
            pid = m.group(1)
            plan_dirs_by_pid[pid].append((d.name, d))
    for pid in plan_dirs_by_pid:
        plan_dirs_by_pid[pid].sort(key=lambda x: x[0])

    total_plan_count = sum(len(v) for v in plan_dirs_by_pid.values())
    if total_plan_count:
        print(f"  Found {total_plan_count} plan() dumps for wrapper class pairing")

    # Process each call dump
    # Collect candidates first, then apply diversity selection before writing.
    candidates: dict[str, list[dict]] = defaultdict(list)
    # Deduplication: track seen axes per definition to avoid redundant workloads.
    # Keep at most 2 entries per unique axes combination (for redundancy in safetensors data).
    seen_axes_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    MAX_DUPS_PER_AXES = 2
    # Collect up to 50x the target so diversity selection has enough candidates
    # across all batch sizes (e.g., 18 rounds × 8 steps × 2 TP = 288 dumps).
    MAX_ENTRIES_PER_DEF = 20
    MAX_CANDIDATES_PER_DEF = MAX_ENTRIES_PER_DEF * 50

    for call_dir in call_dirs:
        # Quick check: what function does this dump contain?
        metadata_path = call_dir / "metadata.jsonl"
        func_name = None
        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        func_name = rec.get("function_name")
                        break
                    except json.JSONDecodeError:
                        continue

        if func_name is None:
            continue

        # Skip plan() dumps themselves — they are consumed as supplements to run() dumps
        if func_name.endswith(".plan"):
            continue

        # Find matching definitions for this function
        matching_defs = func_name_to_defs.get(func_name, [])
        if not matching_defs:
            continue

        # For any wrapper class .run(), find the paired .plan() dump (same pid, most
        # recent plan dir whose name sorts before this run dir's name).
        plan_dir = None
        if func_name.endswith(".run") and plan_dirs_by_pid:
            m = re.search(r"_pid(\d+)_", call_dir.name)
            if m:
                pid = m.group(1)
                plans = plan_dirs_by_pid.get(pid, [])
                matched_plan = None
                for plan_name, plan_path in plans:
                    if plan_name <= call_dir.name:
                        matched_plan = plan_path
                    else:
                        break
                plan_dir = matched_plan

        for def_name in matching_defs:
            # Skip if we already have enough candidates for this definition
            if len(candidates[def_name]) >= MAX_CANDIDATES_PER_DEF:
                continue

            defn = definitions[def_name]

            entry = process_call_dump(
                call_dir, defn, def_name, blob_base_dir, func_name, plan_dir=plan_dir
            )

            if entry is not None:
                # Deduplicate: skip if we already have MAX_DUPS_PER_AXES entries with same axes
                axes_key = json.dumps(entry["workload"]["axes"], sort_keys=True)
                if seen_axes_count[def_name][axes_key] < MAX_DUPS_PER_AXES:
                    seen_axes_count[def_name][axes_key] += 1
                    candidates[def_name].append(entry)

    # Apply diversity selection: pick at most MAX_ENTRIES_PER_DEF entries per definition,
    # preferring diverse shapes. Uses a greedy max-spread algorithm: start with the first
    # entry, then repeatedly pick the candidate that maximises its minimum normalised
    # distance to all already-selected entries across all variable axes.
    def _select_diverse(entries: list[dict], definition: dict, max_count: int) -> list[dict]:
        if len(entries) <= max_count:
            return entries

        var_axes = [
            name for name, spec in definition.get("axes", {}).items() if spec.get("type") == "var"
        ]
        if not var_axes:
            return entries[:max_count]

        def _axes_vec(e: dict) -> list[float]:
            axes = e["workload"]["axes"]
            return [float(axes.get(a, 0)) for a in var_axes]

        def _min_norm_dist(vec: list[float], selected_vecs: list[list[float]]) -> float:
            """Minimum normalised distance between vec and any selected vector."""
            min_d = float("inf")
            for sv in selected_vecs:
                # Normalise each axis by max(abs values) across both to get relative distance
                d = 0.0
                for v1, v2 in zip(vec, sv):
                    norm = max(abs(v1), abs(v2), 1.0)
                    d += ((v1 - v2) / norm) ** 2
                min_d = min(min_d, d**0.5)
            return min_d

        # Phase 1: seed with one representative per distinct batch_size so all
        # batch sizes are guaranteed to appear in the output (when max_count allows).
        # Within each batch_size group, pick the entry with the highest min-distance
        # to the already-selected set (greedy farthest-point within the group).
        from collections import defaultdict as _dd

        by_bs = _dd(list)
        for e in entries:
            bs = e["workload"]["axes"].get("batch_size", 0)
            by_bs[bs].append(e)

        selected: list[dict] = []
        selected_vecs: list[list[float]] = []
        remaining: list[dict] = list(entries)

        for bs in sorted(by_bs):
            if len(selected) >= max_count:
                break
            group = by_bs[bs]
            if not selected_vecs:
                # First entry — just pick the first in the group
                pick = group[0]
            else:
                # Pick the farthest from already-selected within this batch_size group
                pick = max(group, key=lambda e: _min_norm_dist(_axes_vec(e), selected_vecs))
            selected.append(pick)
            selected_vecs.append(_axes_vec(pick))
            remaining.remove(pick)

        # Phase 2: fill remaining slots with greedy farthest-point from all remaining candidates
        while len(selected) < max_count and remaining:
            best_idx = max(
                range(len(remaining)),
                key=lambda i: _min_norm_dist(_axes_vec(remaining[i]), selected_vecs),
            )
            selected.append(remaining[best_idx])
            selected_vecs.append(_axes_vec(remaining[best_idx]))
            remaining.pop(best_idx)

        return selected

    # Apply diversity selection to get final results
    results = defaultdict(list)
    for def_name, entries in candidates.items():
        defn = definitions[def_name]
        selected = _select_diverse(entries, defn, MAX_ENTRIES_PER_DEF)
        results[def_name] = selected
        print(f"  Selected {len(selected)}/{len(entries)} diverse entries for {def_name}")
        for entry in selected:
            print(f"  ✓ {def_name}: {entry['workload']['axes']}")

    # Write output JSONL files
    for def_name, entries in results.items():
        defn = definitions[def_name]
        op_type = defn.get("op_type", "gdn")

        out_dir = flashinfer_trace_dir / "workloads" / op_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{def_name}.jsonl"

        if replace or not out_path.exists():
            mode = "w"
            action = "Replaced" if out_path.exists() else "Created"
        else:
            mode = "a"
            action = "Appended to"

        with open(out_path, mode) as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        # Clean up orphaned blobs AFTER writing the new JSONL so that the
        # cleanup reads the new UUIDs (not the old ones) as the keep-set.
        if replace:
            _cleanup_orphaned_blobs(out_path, flashinfer_trace_dir, def_name, op_type)

        print(f"\n{action} {out_path}: {len(entries)} new workloads")

    return dict(results)


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize FlashInfer tensor dumps into workload JSONL"
    )
    parser.add_argument("--dump-dir", required=True, help="FlashInfer dump directory")
    parser.add_argument(
        "--definitions", nargs="+", help="Definition names (e.g. gdn_mtp_qk4_v8_d128_k_last)"
    )
    parser.add_argument("--op-type", help="Process all definitions of this op_type")
    parser.add_argument(
        "--flashinfer-trace-dir",
        default="~/flashinfer-trace",
        help="Path to flashinfer-trace repo (default: ~/flashinfer-trace)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing workload files instead of appending",
    )
    parser.add_argument(
        "--skip-const-axis-check",
        action="store_true",
        help=(
            "Skip const-axis shape verification. Use when collecting TP=1 SGLang dumps "
            "for a TP=2 definition where head-count axes differ but structural tensors "
            "(indptrs, indices) are identical across TP configurations."
        ),
    )
    args = parser.parse_args()

    global _SKIP_CONST_AXIS_CHECK
    _SKIP_CONST_AXIS_CHECK = args.skip_const_axis_check

    dump_dir = Path(args.dump_dir).expanduser().resolve()
    trace_dir = Path(args.flashinfer_trace_dir).expanduser().resolve()

    if not dump_dir.exists():
        print(f"ERROR: dump directory does not exist: {dump_dir}", file=sys.stderr)
        sys.exit(1)

    if not trace_dir.exists():
        print(f"ERROR: flashinfer-trace directory does not exist: {trace_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect definition files
    def_files = []

    if args.op_type:
        op_type_dir = trace_dir / "definitions" / args.op_type
        if not op_type_dir.exists():
            print(f"ERROR: op_type directory not found: {op_type_dir}", file=sys.stderr)
            sys.exit(1)
        def_files = sorted(op_type_dir.glob("*.json"))
        print(f"Loading all {args.op_type} definitions: {[f.stem for f in def_files]}")

    if args.definitions:
        for def_name in args.definitions:
            # Search across all op_type directories
            found = list((trace_dir / "definitions").glob(f"**/{def_name}.json"))
            if not found:
                print(f"WARNING: definition not found: {def_name}", file=sys.stderr)
            else:
                def_files.extend(found)

    if not def_files:
        print("ERROR: no definition files found", file=sys.stderr)
        sys.exit(1)

    # Deduplicate
    def_files = list({str(f): f for f in def_files}.values())
    print(f"\nSanitizing dumps in: {dump_dir}")
    print(f"Target definitions: {[f.stem for f in def_files]}")
    print(f"Output to: {trace_dir}\n")

    results = sanitize_dumps(dump_dir, def_files, trace_dir, replace=args.replace)

    total = sum(len(v) for v in results.values())
    print(f"\n{'='*60}")
    print(f"Summary: {total} workloads across {len(results)} definitions")
    for def_name, entries in results.items():
        print(f"  {def_name}: {len(entries)} workloads")
    print("=" * 60)


if __name__ == "__main__":
    main()
