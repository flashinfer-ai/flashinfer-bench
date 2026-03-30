"""
Direct workload collection for gqa_paged_decode_h8_kv1_d128_ps64.

Bypasses the SGLang constraint issue by directly calling the FlashInfer
BatchDecodeWithPagedKVCacheWrapper API with synthetic data.
8 q-heads, 1 kv-heads, head_dim=128, page_size=64 (Qwen2.5 72B at TP=8).
"""

import math
import os
import subprocess
import sys
from pathlib import Path

DUMP_DIR = Path("/home/averyh/flashinfer-bench/workload_dumps_gqa_paged_decode_h8kv1d128_ps64")
TRACE_DIR = Path("/home/averyh/flashinfer-bench/tmp/flashinfer-trace")
SANITIZE_SCRIPT = Path("/home/averyh/flashinfer-bench/scripts/sanitize_dumps.py")

# FlashInfer logging env vars must be set before import
os.environ["FLASHINFER_LOGLEVEL"] = "10"
os.environ["FLASHINFER_DUMP_DIR"] = str(DUMP_DIR)
os.environ["FLASHINFER_DUMP_SAFETENSORS"] = "1"
os.environ["FLASHINFER_DUMP_INCLUDE"] = (
    "BatchDecodeWithPagedKVCacheWrapper.run," "BatchDecodeWithPagedKVCacheWrapper.plan"
)
os.environ["FLASHINFER_DUMP_EXCLUDE"] = "*.__init__"
os.environ["FLASHINFER_DUMP_MAX_COUNT"] = "50000"
os.environ["FLASHINFER_DUMP_MAX_SIZE_GB"] = "30"
os.environ["FLASHINFER_USE_CUDA_NORM"] = "1"

DUMP_DIR.mkdir(parents=True, exist_ok=True)

# Ensure we use the full flashinfer package (not namespace stub in site-packages)
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
flashinfer_src = Path("/home/averyh/flashinfer-bench/tmp/flashinfer")
if str(flashinfer_src) not in sys.path:
    sys.path.insert(0, str(flashinfer_src))

import importlib

import flashinfer
import torch

importlib.reload(flashinfer)

NUM_QO_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 64

# (batch_size, avg_kv_len_in_pages) configs to generate diverse workloads
CONFIGS = [
    (1, 50),
    (1, 300),
    (1, 800),
    (2, 50),
    (2, 300),
    (2, 800),
    (4, 50),
    (4, 300),
    (4, 800),
    (8, 50),
    (8, 300),
    (8, 800),
    (16, 50),
    (16, 300),
    (16, 800),
    (32, 50),
    (32, 300),
    (64, 50),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")
fi_version = getattr(flashinfer, "__version__", "unknown")
print(f"FlashInfer version: {fi_version}")

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

for batch_size, avg_kv_pages in CONFIGS:
    try:
        # Generate random page counts around avg_kv_pages
        page_counts = torch.randint(
            max(1, avg_kv_pages // 2), avg_kv_pages * 2 + 1, (batch_size,), dtype=torch.int32
        )
        total_pages = page_counts.sum().item()

        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.cumsum(page_counts.to(device), dim=0)
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
        # last page has some tokens (1 to PAGE_SIZE)
        kv_last_page_len = torch.randint(
            1, PAGE_SIZE + 1, (batch_size,), dtype=torch.int32, device=device
        )

        num_pages = int(total_pages) + 10
        q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        k_cache = torch.randn(
            num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        v_cache = torch.randn(
            num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

        # group_size=8 (8 qo_heads / 1 kv_head) requires use_tensor_cores=True
        decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD", use_tensor_cores=True
        )
        decode_wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            page_size=PAGE_SIZE,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=sm_scale,
        )
        output, lse = decode_wrapper.run(q, (k_cache, v_cache), return_lse=True)

        len_indptr = batch_size + 1
        num_kv_indices = int(kv_indptr[-1].item())
        print(
            f"  OK  batch_size={batch_size:2d}, avg_kv_pages={avg_kv_pages:4d}, "
            f"num_pages={num_pages}, len_indptr={len_indptr}, num_kv_indices={num_kv_indices}"
        )

    except Exception as e:
        print(f"  ERR batch_size={batch_size}, avg_kv_pages={avg_kv_pages}: {e}")

print(f"\nDump dir: {DUMP_DIR}")

# Run sanitize_dumps.py
print("\nRunning sanitize_dumps.py...")
cmd = [
    sys.executable,
    str(SANITIZE_SCRIPT),
    "--dump-dir",
    str(DUMP_DIR),
    "--definitions",
    "gqa_paged_decode_h8_kv1_d128_ps64",
    "--flashinfer-trace-dir",
    str(TRACE_DIR),
]
result = subprocess.run(cmd, capture_output=False)
if result.returncode != 0:
    print("ERROR: sanitize_dumps.py failed", file=sys.stderr)
    sys.exit(1)

print("\nDone!")
