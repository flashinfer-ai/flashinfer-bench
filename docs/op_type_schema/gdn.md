# gdn

Gated Delta Net (GDN) linear attention mechanism. GDN implements the delta rule for linear attention with gated updates, enabling efficient recurrent computation. It supports both GQA (grouped query attention, num_q_heads > num_v_heads) and GVA (grouped value attention, num_v_heads > num_q_heads) configurations.

The delta rule update is:
```
g = -exp(A_log) * softplus(a + dt_bias)   # Log-space decay gate
beta = sigmoid(b)                          # Update gate
state = state * exp(g)                     # Apply decay to state
v_new = v - k @ state                      # Prediction error (delta rule)
v_new = v_new * beta                       # Apply update gate
state = state + k^T @ v_new                # Update state (outer product)
output = scale * q @ state                 # Compute output
```

Variants:
- prefill (chunked computation for variable-length sequences)
- decode (single-token generation with recurrent state update)

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
- The final state is in k-last layout [N, H, V, K]
- Gate tensors (g, beta) are in float32 for numerical stability

## decode

Single-token decoding with recurrent state update. Uses gating parameters (A_log, a, dt_bias, b) to compute decay and update gates. Optionally applies L2 normalization to q and k for numerical stability.

Axes (6 dimensions):
- `batch_size`: variable (number of sequences being decoded)
- `num_q_heads`, `num_k_heads`, `num_v_heads`, `head_size`: constant

Inputs (10 tensors + 1 scalar):
- `q`: query tensor [batch_size, 1, num_q_heads, head_size], bfloat16
- `k`: key tensor [batch_size, 1, num_k_heads, head_size], bfloat16
- `v`: value tensor [batch_size, 1, num_v_heads, head_size], bfloat16
- `state`: recurrent state [batch_size, num_sab_heads, head_size, head_size], float32
- `A_log`: log decay parameter [num_sab_heads], float32
- `a`: input-dependent decay [batch_size, 1, num_sab_heads], bfloat16
- `dt_bias`: decay bias [num_sab_heads], bfloat16
- `b`: update gate input [batch_size, 1, num_sab_heads], bfloat16
- `scale`: scale factor (scalar), float32 (default: 1.0 or 1/sqrt(head_size))
- `use_qk_l2norm`: whether to apply L2 normalization to q and k, bool (default: true)

Outputs (2 tensors):
- `output`: attention output [batch_size, 1, num_o_heads, head_size], bfloat16
- `new_state`: updated recurrent state [batch_size, num_sab_heads, head_size, head_size], float32

Where:
- `num_sab_heads = max(num_q_heads, num_v_heads)` (state and beta heads)
- `num_o_heads = num_v_heads` (output heads follow value heads)

Gating computation:
```
g = -exp(A_log) * softplus(a + dt_bias)   # softplus(x) = log(1 + exp(x))
beta = sigmoid(b)
```

Constraints:
- For GVA: `num_v_heads >= num_q_heads` and `num_v_heads % num_q_heads == 0`
- `num_k_heads == num_q_heads` (keys and queries must have same number of heads)

State Layout:
- **k-last**: [B, H, V, K] - V dimension before K dimension, faster for decode
- **k-first**: [B, H, K, V] - K dimension before V dimension

Notes:
- L2 normalization helps with numerical stability when head_size is large
