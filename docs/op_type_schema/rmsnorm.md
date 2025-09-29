# rmsnorm

 Root Mean Square Layer Normalization (RMSNorm) is a normalization technique that normalizes the input by the root mean square of its elements.

Variants:
- Standard RMSNorm: basic RMS normalization that scales input by RMS and applies learned weight parameters
- Fused Add RMSNorm: adds residual connection before normalization in a single fused operation

Axes (2 dimensions):
- `batch_size`: variable
- `hidden_size`: constant

Inputs (2 or 3 tensors):
- `hidden_states`: [batch_size, hidden_size]
- `weight`: [hidden_size]
- For Fused Add RMSNorm only:
    - `residual`: [batch_size, hidden_size]

Outputs (1 tensor):
- `output`: [batch_size, hidden_size]
