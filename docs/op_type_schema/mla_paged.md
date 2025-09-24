# mla_paged

Variants:
- prefill
- decode

## prefill

Axes (8 dimensions):
- `total_q`, `num_pages`, `len_indptr`, `num_kv_indices`: variable
- `num_qo_heads`, `head_dim_ckv`, `head_dim_kpe`, `page_size`: constant

Inputs (7 tensors + 1 scalar):
- `q_nope`: query tensor without positional encoding [total_q, num_qo_heads, head_dim_ckv]
- `q_pe`: query positional encoding component [total_q, num_qo_heads, head_dim_kpe]
- `ckv_cache`: compressed key-value cache [num_pages, page_size, head_dim_ckv]
- `kpe_cache`: key positional encoding cache [num_pages, page_size, head_dim_kpe]
- `qo_indptr`, `kv_indptr`, `kv_indices`: paging indices
- `sm_scale`: softmax scale (scalar)

Outputs (2 tensors):
- `output`: attention output [total_q, num_qo_heads, head_dim_ckv]
- `lse`: log-sum-exp values [total_q, num_qo_heads]

Constraints:
- `total_q == qo_indptr[-1]`
- `num_kv_indices = kv_indptr[-1]`

## decode

Axes (8 dimensions):
- `batch_size`, `num_pages`, `len_indptr`, `num_kv_indices`: variable
- `num_qo_heads`, `head_dim_ckv`, `head_dim_kpe`, `page_size`: constant

Inputs (6 tensors + 1 scalar):
- `q_nope`: query tensor without positional encoding [batch_size, num_qo_heads, head_dim_ckv]
- `q_pe`: query positional encoding [batch_size, num_qo_heads, head_dim_kpe]
- `ckv_cache`: compressed key-value cache [num_pages, page_size, head_dim_ckv]
- `kpe_cache`: key positional encoding cache [num_pages, page_size, head_dim_kpe]
- `kv_indptr`, `kv_indices`: paging indices
- `sm_scale`: softmax scale (scalar)

Outputs (2 tensors):
- `output`: attention output [batch_size, num_qo_heads, head_dim_ckv]
- `lse`: log-sum-exp values [batch_size, num_qo_heads]

Constraints:
- `len_indptr = num_pages + 1`
- `num_kv_indices = kv_indptr[-1]`
