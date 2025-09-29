# sampling

 Token sampling operations for language model generation. These methods select the next token from a probability distribution, implementing various strategies to balance between diversity and quality in text generation by filtering and sampling from the model's output probabilities.

Variants:
- Top-k sampling: Keeps only the k highest probability tokens, renormalizes the distribution, then samples. Controls diversity by limiting the vocabulary size to the most likely tokens
- Top-p sampling: Filters tokens using cumulative probability threshold (nucleus sampling). Dynamically adjusts vocabulary size based on probability mass, maintaining diversity while avoiding low-probability tokens
- Top-k + Top-p sampling: Combines both filtering methods for fine-grained control over generation quality and diversity

Axes (2 dimensions):
- `batch_size`: variable
- `vocab_size`: constant

Inputs (1 to 3 tensors):
- `probs`: probability distributions after softmax [batch_size, vocab_size]
- Sampling-specific parameters:
  - `top_k`: for top-k sampling [batch_size]
  - `top_p`: for top-p/nucleus sampling [batch_size]

Outputs (1 tensor):
- `samples`: sampled token indices [batch_size]
