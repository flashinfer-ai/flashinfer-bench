## **gqa_paged_prefill**

**Axes (8 dimensions):**
- `total_q`, `num_pages`, `len_indptr`, `num_kv_indices`: variable
- `num_qo_heads`, `num_kv_heads`, `head_dim`, `page_size`: constant

**Inputs (6 tensors + 1 scalar):**
- `q`: query tensor [total_q, num_qo_heads, head_dim]
- `k_cache`, `v_cache`: paged KV cache [num_pages, page_size, num_kv_heads, head_dim]
- `qo_indptr`, `kv_indptr`, `kv_indices`: paging indices
- `sm_scale`: softmax scale (scalar)

**Outputs:**
- `output`: attention output [total_q, num_qo_heads, head_dim]
- `lse`: log-sum-exp values [total_q, num_qo_heads]

**Constraints:**
- `total_q == qo_indptr[-1]`
- `num_kv_indices = kv_indptr[-1]`
