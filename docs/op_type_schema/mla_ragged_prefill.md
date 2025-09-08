## **mla_ragged_prefill**

**Axes (5 dimensions):**
- `total_tokens`, `len_indptr`: variable
- `num_qo_heads`, `head_dim_qk`, `head_dim_vo`: constant

**Inputs (4 tensors + 1 scalar):**
- `q`: query tensor [total_tokens, num_qo_heads, head_dim_qk]
- `k`, `v`: key-value tensors [total_tokens, num_qo_heads, head_dim_vo]
- `seq_indptr`: sequence offsets
- `sm_scale`: softmax scale (scalar)

**Outputs:**
- `output`: attention output [total_tokens, num_qo_heads, head_dim]
- `lse`: log-sum-exp values [total_tokens, num_qo_heads]

**Constraints:**
- `total_tokens == seq_indptr[-1]`
