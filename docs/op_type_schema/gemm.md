## **gemm**

**Axes (3 dimensions):**
- `M`: variable dimension
- `N`: constant (must specify value in implementation)
- `K`: constant (must specify value in implementation)

**Inputs (2 tensors):**
- `A`: shape [M, K]
- `B`: shape [N, K]

**Outputs:**
- `C`: shape [M, N]
