## **gqa_paged_decode**

**Axes (8 dimensions):**
- `batch_size`, `num_pages`, `len_indptr`, `num_kv_indices`: variable
- `num_qo_heads`, `num_kv_heads`, `head_dim`, `page_size`: constant

**Inputs (5 tensors + 1 scalar):**
- `q`: query tensor [batch_size, num_qo_heads, head_dim]
- `k_cache`, `v_cache`: paged KV cache [num_pages, page_size, num_kv_heads, head_dim]
- `kv_indptr`, `kv_indices`: paging indices
- `sm_scale`: softmax scale (scalar)

**Outputs:**
- `output`: attention output [batch_size, num_qo_heads, head_dim]
- `lse`: log-sum-exp values [batch_size, num_qo_heads]

**Constraints:**
- `len_indptr = num_pages + 1`
- `num_kv_indices = kv_indptr[-1]`
