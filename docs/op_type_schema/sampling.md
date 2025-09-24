# sampling

Variants:
- Top-k sampling: Keeps only the k highest probability tokens, renormalizes, then samples
- Top-p sampling: Filters using cumulative probability threshold (nucleus sampling)
- Top-k + Top-p sampling: Combines both filtering methods

Axes (2 dimensions):
- `batch_size`: variable
- `vocab_size`: constant

Inputs (1 to 3 tensors):
- `probs`: probability distributions after softmax [batch_size, vocab_size], dtype: float32
- Sampling-specific parameters:
  - `top_k`: for top-k sampling [batch_size], dtype: int32
  - `top_p`: for top-p/nucleus sampling [batch_size], dtype: float32

Outputs (1 tensor):
- `samples`: sampled token indices [batch_size], dtype: int64
