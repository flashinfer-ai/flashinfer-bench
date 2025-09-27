# gqa_paged

Grouped Query Attention (GQA) with paged memory layout. GQA is an attention mechanism that reduces memory usage by grouping multiple query heads to share the same key-value heads, while using a paged memory system for efficient KV cache management. This allows for variable-length sequences and better memory utilization compared to traditional attention.

Variants:
- prefill
- decode

## prefill

Axes (8 dimensions):
- `total_q`, `num_pages`, `len_indptr`, `num_kv_indices`: variable
- `num_qo_heads`, `num_kv_heads`, `head_dim`, `page_size`: constant

Inputs (6 tensors + 1 scalar):
- `q`: query tensor [total_q, num_qo_heads, head_dim]
- `k_cache`, `v_cache`: paged KV cache [num_pages, page_size, num_kv_heads, head_dim]
- `qo_indptr`, `kv_indptr`, `kv_indices`: paging indices
- `sm_scale`: softmax scale (scalar)

Outputs (2 tensors):
- `output`: attention output [total_q, num_qo_heads, head_dim]
- `lse`: log-sum-exp values [total_q, num_qo_heads]

Constraints:
- `total_q == qo_indptr[-1]`
- `num_kv_indices = kv_indptr[-1]`

## decode

Axes (8 dimensions):
- `batch_size`, `num_pages`, `len_indptr`, `num_kv_indices`: variable
- `num_qo_heads`, `num_kv_heads`, `head_dim`, `page_size`: constant

Inputs (5 tensors + 1 scalar):
- `q`: query tensor [batch_size, num_qo_heads, head_dim]
- `k_cache`, `v_cache`: paged KV cache [num_pages, page_size, num_kv_heads, head_dim]
- `kv_indptr`, `kv_indices`: paging indices
- `sm_scale`: softmax scale (scalar)

Outputs (2 tensors):
- `output`: attention output [batch_size, num_qo_heads, head_dim]
- `lse`: log-sum-exp values [batch_size, num_qo_heads]

Constraints:
- `len_indptr = num_pages + 1`
- `num_kv_indices = kv_indptr[-1]`
