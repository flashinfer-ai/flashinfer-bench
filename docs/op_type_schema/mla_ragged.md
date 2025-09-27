# mla_ragged

## prefill

Axes (5 dimensions):
- `total_tokens`, `len_indptr`: variable
- `num_qo_heads`, `head_dim_qk`, `head_dim_vo`: constant

Inputs (4 tensors + 1 scalar):
- `q`: query tensor [total_tokens, num_qo_heads, head_dim_qk]
- `k`: key tensors [total_tokens, num_qo_heads, head_dim_qk]
- `v`: value tensors [total_tokens, num_qo_heads, head_dim_vo]
- `seq_indptr`: sequence offsets
- `sm_scale`: softmax scale (scalar)

Outputs (2 tensors):
- `output`: attention output [total_tokens, num_qo_heads, head_dim_vo]
- `lse`: log-sum-exp values [total_tokens, num_qo_heads]

Constraints:
- `total_tokens == seq_indptr[-1]`
