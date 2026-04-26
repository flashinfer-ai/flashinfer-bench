#!/usr/bin/env python3
"""
Post-process FlashInfer plan-only dumps from a sglang real-workload run into
canonical flashinfer-trace workload JSONL entries.

Why a separate tool? sanitize_dumps.py treats .plan() dumps as supplements to
.run() dumps; with .run() excluded from FLASHINFER_DUMP_INCLUDE (because the
captured ckv_cache D2H copy on every cuda-graph replay tanks decode
throughput by ~2000x), the orthodox sanitize path produces zero entries.

Plan() takes the small structural tensors (qo_indptr, kv_indptr, kv_indices,
kv_lens) — exactly what the canonical workload JSONL stores as safetensors;
the large q / ckv_cache / kpe_cache tensors are stored as "type": "random"
in the schema (no values needed), so we can synthesize entries from .plan()
dumps + the definition's static axes + a runtime num_pages constant.

Usage:
    python postprocess_plan_dumps.py \\
        --dump-dir /tmp/fi_dumps_dsr1 \\
        --definition mla_paged_decode_h16_ckv512_kpe64_ps1 \\
        --num-pages 1062933 \\
        --trace-dir ~/flashinfer-trace \\
        --subdir inferencex
"""

import argparse
import json
import shutil
import uuid
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument(
        "--definition", required=True, help="e.g. mla_paged_decode_h16_ckv512_kpe64_ps1"
    )
    ap.add_argument(
        "--num-pages",
        required=True,
        type=int,
        help="from sglang's 'KV Cache is allocated. #tokens=N' line",
    )
    ap.add_argument("--trace-dir", required=True, type=Path)
    ap.add_argument(
        "--subdir",
        default="inferencex",
        help="namespace under workloads/ to keep the run isolated from canonical entries",
    )
    ap.add_argument(
        "--max-per-shape", type=int, default=2, help="dedup per (batch_size, num_kv_indices)"
    )
    args = ap.parse_args()

    args.trace_dir = args.trace_dir.expanduser().resolve()
    args.dump_dir = args.dump_dir.expanduser().resolve()

    # Locate definition + op_type
    def_path = next(args.trace_dir.glob(f"definitions/**/{args.definition}.json"), None)
    if def_path is None:
        raise SystemExit(f"definition not found: {args.definition}")
    op_type = def_path.parent.name
    defn = json.loads(def_path.read_text())

    # Output paths under inferencex/ namespace
    out_jsonl = args.trace_dir / "workloads" / args.subdir / op_type / f"{args.definition}.jsonl"
    out_blob_dir = args.trace_dir / "blob" / "workloads" / args.subdir / op_type / args.definition
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_blob_dir.mkdir(parents=True, exist_ok=True)

    # Walk plan dumps. Each shape (batch_size, num_kv_indices) seen up to
    # --max-per-shape times for redundancy.
    plan_dirs = sorted(p for p in args.dump_dir.iterdir() if p.is_dir() and ".plan_" in p.name)
    print(f"scanning {len(plan_dirs)} plan dump dirs")

    shape_count: dict[tuple, int] = defaultdict(int)
    entries = []

    from safetensors.torch import load_file, save_file

    for d in plan_dirs:
        meta_path = d / "metadata.jsonl"
        st_path = d / "inputs.safetensors"
        if not meta_path.exists() or not st_path.exists():
            continue
        try:
            tensors = load_file(str(st_path))
        except Exception as exc:
            print(f"  skip {d.name}: {exc}")
            continue

        # plan() signature: arg_0=self, arg_1=qo_indptr, arg_2=kv_indptr, arg_3=kv_indices, arg_4=kv_lens, ...
        if not all(k in tensors for k in ("arg_2", "arg_3")):
            continue
        kv_indptr = tensors["arg_2"]
        kv_indices = tensors["arg_3"]
        batch_size = int(kv_indptr.shape[0]) - 1  # indptr length = batch + 1
        if batch_size <= 0:
            continue
        num_kv_indices = int(kv_indices.shape[0])
        len_indptr = int(kv_indptr.shape[0])

        # Dedup by exact shape signature
        sig = (batch_size, num_kv_indices)
        if shape_count[sig] >= args.max_per_shape:
            continue
        shape_count[sig] += 1

        # Save kv_indptr + kv_indices as a single safetensors blob
        u = str(uuid.uuid4())
        blob_name = f"{args.definition}_{u}.safetensors"
        blob_path = out_blob_dir / blob_name
        save_file(
            {"kv_indptr": kv_indptr.contiguous(), "kv_indices": kv_indices.contiguous()},
            str(blob_path),
        )

        # Build canonical workload entry
        entry = {
            "definition": args.definition,
            "solution": None,
            "workload": {
                "uuid": u,
                "axes": {
                    "batch_size": batch_size,
                    "num_pages": args.num_pages,
                    "len_indptr": len_indptr,
                    "num_kv_indices": num_kv_indices,
                },
                "inputs": {
                    "q_nope": {"type": "random"},
                    "q_pe": {"type": "random"},
                    "ckv_cache": {"type": "random"},
                    "kpe_cache": {"type": "random"},
                    "sm_scale": {"type": "scalar", "value": 0.08838834764831843},
                    "kv_indptr": {
                        "type": "safetensors",
                        "path": f"./blob/workloads/{args.subdir}/{op_type}/{args.definition}/{blob_name}",
                        "tensor_key": "kv_indptr",
                    },
                    "kv_indices": {
                        "type": "safetensors",
                        "path": f"./blob/workloads/{args.subdir}/{op_type}/{args.definition}/{blob_name}",
                        "tensor_key": "kv_indices",
                    },
                },
            },
            "evaluation": None,
        }
        entries.append(entry)

    # Write JSONL
    with open(out_jsonl, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"wrote {len(entries)} entries to {out_jsonl}")
    print(f"unique (batch_size, num_kv_indices) shapes: {len(shape_count)}")
    for (bs, nki), c in sorted(shape_count.items()):
        print(f"  batch={bs} num_kv_indices={nki}: kept {min(c, args.max_per_shape)}")


if __name__ == "__main__":
    main()
