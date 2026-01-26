"""
Tests comparing FlashInfer kernels against flashinfer_bench definition references.

This test directly loads and runs the reference implementations from definition JSON files
using BuilderRegistry.build_reference(), instead of duplicating the reference logic.
"""

import math
from pathlib import Path

import pytest
import torch

try:
    import flashinfer

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

try:
    import deep_gemm

    DEEP_GEMM_AVAILABLE = True
except ImportError:
    DEEP_GEMM_AVAILABLE = False

try:
    from flashinfer_bench.compile.registry import BuilderRegistry
    from flashinfer_bench.data.trace_set import TraceSet

    FLASHINFER_BENCH_AVAILABLE = True
except ImportError:
    FLASHINFER_BENCH_AVAILABLE = False

# Path to definitions
TRACE_ROOT = Path(__file__).resolve().parents[2]

# DeepSeek MLA parameters
NUM_QO_HEADS = 16
QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 256

# FP8 Indexer parameters (deep_gemm requirements)
FP8_NUM_INDEX_HEADS = 64
FP8_INDEX_HEAD_DIM = 128


def load_definition(name: str):
    """Load a definition from the trace set."""
    trace_set = TraceSet.from_path(TRACE_ROOT)
    return trace_set.definitions[name]


def build_reference_runnable(definition):
    """Build the reference implementation as a runnable."""
    registry = BuilderRegistry.get_instance()
    return registry.build_reference(definition)


# ============================================================================
# DSA Sparse Decode Tests
# ============================================================================


@pytest.mark.skipif(not FLASHINFER_AVAILABLE, reason="FlashInfer not available")
@pytest.mark.skipif(not FLASHINFER_BENCH_AVAILABLE, reason="flashinfer_bench not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trtllm_mla_sparse_vs_definition_reference():
    """
    Test FlashInfer trtllm MLA sparse against dsa_sparse_decode definition reference.
    """
    torch.manual_seed(42)
    device = "cuda"

    # Load definition and build reference
    definition = load_definition("dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps64")
    reference = build_reference_runnable(definition)

    print(f"\nLoaded definition: {definition.name}")
    print(f"Description: {definition.description}")

    batch_size = 4
    max_seq_len = 1024
    max_num_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = batch_size * max_num_pages
    total_tokens = num_pages * PAGE_SIZE

    # Generate inputs matching definition schema
    q_nope = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    sparse_indices = torch.randint(
        0, total_tokens, (batch_size, TOPK), dtype=torch.int32, device=device
    )
    sm_scale = torch.tensor(
        1.0 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), dtype=torch.float32, device=device
    )

    # Run definition reference
    print("\nRunning definition reference...")
    ref_output, ref_lse = reference(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

    # Prepare FlashInfer inputs (trtllm-gen format)
    query = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)  # [batch, 1, heads, 576]
    kv_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)  # [num_pages, page_size, 576]
    block_tables = sparse_indices.unsqueeze(1)  # [batch, 1, topk]
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    seq_lens = torch.full((batch_size,), total_tokens, dtype=torch.int32, device=device)

    # bmm1_scale = sm_scale (trtllm uses it directly)
    bmm1_scale = 1.0 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)

    # Run FlashInfer
    print("Running FlashInfer trtllm MLA sparse...")
    fi_output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=total_tokens,
        sparse_mla_top_k=TOPK,
        bmm1_scale=bmm1_scale,
    )
    fi_output = fi_output.squeeze(1)  # [batch, heads, 512]

    # Compare
    print("\nComparing outputs...")
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.float().flatten(), fi_output.float().flatten(), dim=0
    ).item()
    max_diff = (ref_output.float() - fi_output.float()).abs().max().item()
    mean_diff = (ref_output.float() - fi_output.float()).abs().mean().item()

    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")

    atol, rtol = 1e-2, 5e-2
    allclose = torch.allclose(ref_output.float(), fi_output.float(), atol=atol, rtol=rtol)

    if allclose:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        left = (ref_output.float() - fi_output.float()).abs()
        right = atol + rtol * ref_output.float().abs()
        hit_ratio = (left <= right).float().mean().item()
        print(f"\nHit ratio: {hit_ratio:.4f}")
        assert hit_ratio >= 0.85, f"Hit ratio {hit_ratio:.4f} below 85%"
        print(f"✓ PASSED: Hit ratio {hit_ratio:.4f} >= 85%")


# ============================================================================
# DSA TopK Indexer Tests (FP8 with deep_gemm + FlashInfer)
# ============================================================================


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    """Convert KV cache to FP8 format (deep_gemm format).

    Input: [num_blocks, block_size, num_heads, head_dim]
    Output: [num_blocks, block_size, num_heads, head_dim + 4] int8 (interpreted as uint8)
    Memory layout: all FP8 values first, then all scales
    """
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(
        dtype=torch.uint8
    )
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(dtype=torch.uint8)
    # Return as int8 to match definition schema (bit pattern is identical)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4).view(torch.int8)


@pytest.mark.skipif(not FLASHINFER_AVAILABLE, reason="FlashInfer not available")
@pytest.mark.skipif(not DEEP_GEMM_AVAILABLE, reason="deep_gemm not available")
@pytest.mark.skipif(not FLASHINFER_BENCH_AVAILABLE, reason="flashinfer_bench not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_topk_indexer_fp8_vs_definition_reference():
    """
    Test deep_gemm FP8 scores + FlashInfer top-k against FP8 definition reference.

    Pipeline: deep_gemm.fp8_paged_mqa_logits -> flashinfer.top_k_page_table_transform
    """
    torch.manual_seed(42)
    device = "cuda"

    # Load FP8 definition and build reference
    definition = load_definition("dsa_topk_indexer_fp8_h64_d128_topk256_ps64")
    reference = build_reference_runnable(definition)

    print(f"\nLoaded definition: {definition.name}")
    print(f"Description: {definition.description}")

    batch_size = 4
    max_seq_len = 1024
    max_num_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = batch_size * max_num_pages + 10

    # Generate random data in bf16, then quantize to FP8
    q_bf16 = torch.randn(
        batch_size, FP8_NUM_INDEX_HEADS, FP8_INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    k_bf16 = torch.randn(
        num_pages, PAGE_SIZE, 1, FP8_INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    # Quantize to FP8
    q_index_fp8 = q_bf16.to(torch.float8_e4m3fn)
    k_index_cache_fp8 = kv_cache_cast_to_fp8(k_bf16)  # [num_pages, page_size, 1, 132]

    # Random weights
    weights = torch.randn(batch_size, FP8_NUM_INDEX_HEADS, dtype=torch.float32, device=device)

    # Sequence lengths and block table
    min_len = TOPK
    seq_lens = torch.randint(
        min_len, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )

    block_table = torch.zeros(batch_size, max_num_pages, dtype=torch.int32, device=device)
    page_offset = 0
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_table[b, :num_pages_for_seq] = torch.arange(
            page_offset, page_offset + num_pages_for_seq, dtype=torch.int32, device=device
        )
        page_offset += num_pages_for_seq

    # Run definition reference
    print("\nRunning definition reference...")
    ref_result = reference(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
    ref_indices = ref_result

    # Run deep_gemm to compute FP8 scores (deep_gemm expects uint8)
    # deep_gemm expects q shape: [batch, next_n, heads, head_dim]
    print("Running deep_gemm.fp8_paged_mqa_logits...")
    q_index_fp8_4d = q_index_fp8.unsqueeze(1)  # [batch, 1, heads, head_dim]
    k_index_cache_uint8 = k_index_cache_fp8.view(torch.uint8)
    max_context_len = max_num_pages * PAGE_SIZE
    # Get schedule metadata for deep_gemm
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(seq_lens, PAGE_SIZE, num_sms)
    logits = deep_gemm.fp8_paged_mqa_logits(
        q_index_fp8_4d,
        k_index_cache_uint8,
        weights,
        seq_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits=False,
    )

    # Build token-level page table for FlashInfer
    token_page_table = torch.zeros(
        batch_size, max_num_pages * PAGE_SIZE, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        for p in range(num_pages_for_seq):
            page_idx = block_table[b, p].item()
            for t in range(PAGE_SIZE):
                token_idx = p * PAGE_SIZE + t
                if token_idx < seq_len:
                    token_page_table[b, token_idx] = page_idx * PAGE_SIZE + t

    # Run FlashInfer top_k_page_table_transform
    print("Running FlashInfer top_k_page_table_transform...")
    fi_indices = flashinfer.top_k_page_table_transform(
        input=logits.to(torch.float16), src_page_table=token_page_table, lengths=seq_lens, k=TOPK
    )

    # Compare indices (order may differ, compare as sets)
    print("\nComparing results...")
    total_match = 0
    total_count = 0
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        actual_topk = min(TOPK, seq_len)

        ref_set = set(ref_indices[b, :actual_topk].tolist())
        fi_set = set(fi_indices[b, :actual_topk].tolist())
        ref_set.discard(-1)
        fi_set.discard(-1)

        intersection = len(ref_set & fi_set)
        total_match += intersection
        total_count += len(ref_set)

        print(f"  Batch {b}: overlap={intersection}/{len(ref_set)}")

    recall = total_match / total_count if total_count > 0 else 1.0
    print(f"\nOverall recall: {recall:.4f} ({total_match}/{total_count})")

    assert recall >= 0.99, f"Recall {recall:.4f} below 99%"
    print("✓ PASSED: deep_gemm + FlashInfer matches FP8 definition reference")


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.skipif(not FLASHINFER_AVAILABLE, reason="FlashInfer not available")
@pytest.mark.skipif(not FLASHINFER_BENCH_AVAILABLE, reason="flashinfer_bench not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("max_seq_len", [512, 1024, 2048])
def test_trtllm_mla_sparse_various_configs(batch_size, max_seq_len):
    """Test with various batch sizes and sequence lengths."""
    torch.manual_seed(42)
    device = "cuda"

    definition = load_definition("dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps64")
    reference = build_reference_runnable(definition)

    max_num_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = batch_size * max_num_pages
    total_tokens = num_pages * PAGE_SIZE

    q_nope = torch.randn(
        batch_size, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    sparse_indices = torch.randint(
        0, total_tokens, (batch_size, TOPK), dtype=torch.int32, device=device
    )
    sm_scale = torch.tensor(
        1.0 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), dtype=torch.float32, device=device
    )

    ref_output, ref_lse = reference(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

    query = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)
    kv_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)
    block_tables = sparse_indices.unsqueeze(1)
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    seq_lens = torch.full((batch_size,), total_tokens, dtype=torch.int32, device=device)
    bmm1_scale = 1.0 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)

    fi_output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=total_tokens,
        sparse_mla_top_k=TOPK,
        bmm1_scale=bmm1_scale,
    )
    fi_output = fi_output.squeeze(1)

    atol, rtol = 1e-2, 5e-2
    allclose = torch.allclose(ref_output.float(), fi_output.float(), atol=atol, rtol=rtol)
    if not allclose:
        left = (ref_output.float() - fi_output.float()).abs()
        right = atol + rtol * ref_output.float().abs()
        hit_ratio = (left <= right).float().mean().item()
        assert hit_ratio >= 0.85, f"Hit ratio {hit_ratio:.4f} below 85%"


@pytest.mark.skipif(not FLASHINFER_AVAILABLE, reason="FlashInfer not available")
@pytest.mark.skipif(not DEEP_GEMM_AVAILABLE, reason="deep_gemm not available")
@pytest.mark.skipif(not FLASHINFER_BENCH_AVAILABLE, reason="flashinfer_bench not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("max_seq_len", [512, 1024, 2048])
def test_topk_indexer_fp8_various_configs(batch_size, max_seq_len):
    """Test FP8 topk indexer with various configurations."""
    torch.manual_seed(42)
    device = "cuda"

    definition = load_definition("dsa_topk_indexer_fp8_h64_d128_topk256_ps64")
    reference = build_reference_runnable(definition)

    max_num_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = batch_size * max_num_pages + 10

    q_bf16 = torch.randn(
        batch_size, FP8_NUM_INDEX_HEADS, FP8_INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    k_bf16 = torch.randn(
        num_pages, PAGE_SIZE, 1, FP8_INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    q_index_fp8 = q_bf16.to(torch.float8_e4m3fn)
    k_index_cache_fp8 = kv_cache_cast_to_fp8(k_bf16)

    weights = torch.randn(batch_size, FP8_NUM_INDEX_HEADS, dtype=torch.float32, device=device)

    min_len = TOPK
    seq_lens = torch.randint(
        min_len, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )

    block_table = torch.zeros(batch_size, max_num_pages, dtype=torch.int32, device=device)
    page_offset = 0
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_table[b, :num_pages_for_seq] = torch.arange(
            page_offset, page_offset + num_pages_for_seq, dtype=torch.int32, device=device
        )
        page_offset += num_pages_for_seq

    ref_result = reference(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
    ref_indices = ref_result

    q_index_fp8_4d = q_index_fp8.unsqueeze(1)
    k_index_cache_uint8 = k_index_cache_fp8.view(torch.uint8)
    max_context_len = max_num_pages * PAGE_SIZE
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(seq_lens, PAGE_SIZE, num_sms)
    logits = deep_gemm.fp8_paged_mqa_logits(
        q_index_fp8_4d,
        k_index_cache_uint8,
        weights,
        seq_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits=False,
    )

    token_page_table = torch.zeros(
        batch_size, max_num_pages * PAGE_SIZE, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        for p in range(num_pages_for_seq):
            page_idx = block_table[b, p].item()
            for t in range(PAGE_SIZE):
                token_idx = p * PAGE_SIZE + t
                if token_idx < seq_len:
                    token_page_table[b, token_idx] = page_idx * PAGE_SIZE + t

    fi_indices = flashinfer.top_k_page_table_transform(
        input=logits.to(torch.float16), src_page_table=token_page_table, lengths=seq_lens, k=TOPK
    )

    total_match = 0
    total_count = 0
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        actual_topk = min(TOPK, seq_len)
        ref_set = set(ref_indices[b, :actual_topk].tolist())
        fi_set = set(fi_indices[b, :actual_topk].tolist())
        ref_set.discard(-1)
        fi_set.discard(-1)
        total_match += len(ref_set & fi_set)
        total_count += len(ref_set)

    recall = total_match / total_count if total_count > 0 else 1.0
    assert recall >= 0.99, f"Recall {recall:.4f} below 99%"


def main():
    """Run tests manually."""
    print("=" * 70)
    print("Testing FlashInfer vs Definition References")
    print("=" * 70)

    if not FLASHINFER_AVAILABLE:
        print("SKIPPED: FlashInfer not available")
        return

    if not DEEP_GEMM_AVAILABLE:
        print("SKIPPED: deep_gemm not available")
        return

    if not FLASHINFER_BENCH_AVAILABLE:
        print("SKIPPED: flashinfer_bench not available")
        return

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available")
        return

    tests = [
        (
            "trtllm MLA sparse vs definition reference",
            test_trtllm_mla_sparse_vs_definition_reference,
        ),
        ("FP8 topk indexer vs definition reference", test_topk_indexer_fp8_vs_definition_reference),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print("=" * 70)
        try:
            test_fn()
            results.append((name, True))
        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    for name, passed in results:
        print(f"  {name}: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
