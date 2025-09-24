# gqa_ragged

## prefill

Axes (6 dimensions):
- `total_q`, `total_kv`, `len_indptr`: variable
- `num_qo_heads`, `num_kv_heads`, `head_dim`: constant

Inputs (5 tensors + 1 scalar):
- `q`: query tensor [total_q, num_qo_heads, head_dim]
- `k`, `v`: key-value tensors [total_kv, num_kv_heads, head_dim]
- `qo_indptr`, `kv_indptr`: sequence offsets
- `sm_scale`: softmax scale (scalar)

Outputs (2 tensors):
- `output`: attention output [total_q, num_qo_heads, head_dim]
- `lse`: log-sum-exp values [total_q, num_qo_heads]

Constraints:
- `total_q == qo_indptr[-1]`
- `total_kv == kv_indptr[-1]`
