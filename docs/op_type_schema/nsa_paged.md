# nsa_paged

Native Sparse Attention (NSA) with paged memory layout. NSA is a sparse attention mechanism used in DeepSeek-V3/R1 that selects only the top-K most relevant KV cache entries for attention computation, significantly reducing memory bandwidth and compute requirements for long sequences. It builds on MLA (Multi-head Latent Attention) with compressed KV and positional encoding components.

The sparse attention workflow consists of:
1. **Indexer**: Computes attention scores and selects top-K KV cache indices
2. **Sparse Attention**: Performs attention only on the selected sparse indices

Variants:
- prefill (with causal masking)
- decode
- indexer (top-K selection)

## prefill

Axes (7 dimensions):
- `total_num_tokens`, `num_pages`: variable
- `num_qo_heads`, `head_dim_ckv`, `head_dim_kpe`, `page_size`, `topk`: constant

Inputs (5 tensors + 1 scalar):
- `q_nope`: query tensor without positional encoding [total_num_tokens, num_qo_heads, head_dim_ckv]
- `q_pe`: query positional encoding component [total_num_tokens, num_qo_heads, head_dim_kpe]
- `ckv_cache`: compressed key-value cache [num_pages, page_size, head_dim_ckv]
- `kpe_cache`: key positional encoding cache [num_pages, page_size, head_dim_kpe]
- `sparse_indices`: top-K KV cache indices per token [total_num_tokens, topk], int32 (-1 for padding)
- `sm_scale`: softmax scale (scalar)

Outputs (2 tensors):
- `output`: attention output [total_num_tokens, num_qo_heads, head_dim_ckv]
- `lse`: 2-based log-sum-exp values [total_num_tokens, num_qo_heads]

Constraints:
- `sparse_indices.shape[0] == total_num_tokens`
- `sparse_indices.shape[-1] == topk`
- `ckv_cache.shape[1] == page_size`

Notes:
- Causal masking is applied implicitly through the sparse_indices (only valid past tokens are indexed)
- Values of -1 in sparse_indices indicate padding (invalid indices)
- Uses MLA-style split query (q_nope + q_pe) for positional encoding

## decode

Axes (7 dimensions):
- `batch_size`, `num_pages`: variable
- `num_qo_heads`, `head_dim_ckv`, `head_dim_kpe`, `page_size`, `topk`: constant

Inputs (5 tensors + 1 scalar):
- `q_nope`: query tensor without positional encoding [batch_size, num_qo_heads, head_dim_ckv]
- `q_pe`: query positional encoding component [batch_size, num_qo_heads, head_dim_kpe]
- `ckv_cache`: compressed key-value cache [num_pages, page_size, head_dim_ckv]
- `kpe_cache`: key positional encoding cache [num_pages, page_size, head_dim_kpe]
- `sparse_indices`: top-K KV cache indices per batch [batch_size, topk], int32 (-1 for padding)
- `sm_scale`: softmax scale (scalar)

Outputs (2 tensors):
- `output`: attention output [batch_size, num_qo_heads, head_dim_ckv]
- `lse`: 2-based log-sum-exp values [batch_size, num_qo_heads]

Constraints:
- `sparse_indices.shape[-1] == topk`
- `ckv_cache.shape[1] == page_size`

## indexer

The indexer stage selects which KV cache entries to attend to based on attention scores with a low-rank query/key representation.

Axes (7 dimensions):
- `batch_size`, `max_seq_len`, `num_pages`: variable
- `num_index_heads`, `index_head_dim`, `page_size`, `topk`: constant

Inputs (4 tensors):
- `q_index`: query tensor for indexing [batch_size, num_index_heads, index_head_dim]
- `k_index_cache`: key index cache [num_pages, page_size, index_head_dim]
- `seq_lens`: sequence lengths [batch_size], int32
- `page_table`: page table mapping positions to pages [batch_size, max_seq_len], int32

Outputs (2 tensors):
- `topk_indices`: selected top-K page indices [batch_size, topk], int32 (-1 for padding)
- `topk_scores`: attention scores for selected indices [batch_size, topk], float32

Constraints:
- `topk <= max_seq_len`

Notes:
- The indexer uses a separate low-rank representation for efficient scoring
- Scores are averaged across heads before top-K selection
- Output indices can be used directly as sparse_indices for the attention kernels

## DeepSeek-V3 Configuration

For DeepSeek-V3 with tensor parallel size 8:
- `num_qo_heads = 16` (128 total / 8 TP)
- `head_dim_ckv = 512` (compressed KV dimension)
- `head_dim_kpe = 64` (positional encoding dimension)
- `page_size = 1` (token-level paging)
- `topk = 256` (sparse selection count)
- `sm_scale = 1/sqrt(192)` (based on pre-absorption head dim: 128 + 64)
