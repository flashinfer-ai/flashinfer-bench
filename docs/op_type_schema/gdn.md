# gdn

Gated Delta Net (GDN) linear attention mechanism. GDN implements the delta rule for linear attention with gated updates, enabling efficient recurrent computation. It supports both GQA (grouped query attention, num_q_heads > num_v_heads) and GVA (grouped value attention, num_v_heads > num_q_heads) configurations.

The delta rule update is:
```
state_new = alpha * state_old - k^T * (k @ state_old) + k^T * (beta * v + (1-beta) * k @ state_old)
output = scale * q @ state_new
```

Variants:
- prefill (chunked computation for variable-length sequences)

## prefill

Axes (6 dimensions):
- `total_seq_len`, `num_seqs`: variable
- `num_q_heads`, `num_k_heads`, `num_v_heads`, `head_size`: constant

Inputs (7 tensors + 1 scalar):
- `q`: query tensor [total_seq_len, num_q_heads, head_size]
- `k`: key tensor [total_seq_len, num_k_heads, head_size]
- `v`: value tensor [total_seq_len, num_v_heads, head_size]
- `g`: forget gate (alpha) [total_seq_len, num_sab_heads], float32, optional (defaults to ones)
- `beta`: update gate [total_seq_len, num_sab_heads], float32, optional (defaults to ones)
- `cu_seqlens`: cumulative sequence lengths [num_seqs + 1], int64
- `initial_state`: initial KV state [num_seqs, num_sab_heads, head_size, head_size], float32, optional
- `scale`: softmax scale (scalar), optional (defaults to 1/sqrt(head_size))

Outputs (2 tensors):
- `output`: attention output [total_seq_len, num_o_heads, head_size]
- `final_state`: final KV state [num_seqs, num_sab_heads, head_size, head_size], float32

Where:
- `num_sab_heads = max(num_q_heads, num_v_heads)` (state and beta heads)
- `num_o_heads = max(num_q_heads, num_v_heads)` (output heads)

Constraints:
- `total_seq_len == cu_seqlens[-1]`
- `num_seqs == len(cu_seqlens) - 1`
- For GQA: `num_q_heads >= num_k_heads` and `num_q_heads % num_k_heads == 0`
- For GVA: `num_v_heads >= num_q_heads` and `num_v_heads % num_q_heads == 0`
- `num_k_heads == num_v_heads` (keys and values must have same number of heads)

Notes:
- Requires SM90 (Hopper) architecture for FlashInfer kernel
- The final state is in k-major layout [N, H, K, V]
- Gate tensors (g, beta) are in float32 for numerical stability
