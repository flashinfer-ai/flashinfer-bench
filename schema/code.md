# Code Format

The code should be a Python file, containing a global function `forward`, any other Python functions,
and any kernel code.

## Kernel Code

The kernel code can be in
* CUDA
* Triton
* TVM
* torch.compile
* etc.

The code should handle the linking between the Python code and the kernel code.

## Parameters of `forward`

The parameters of `forward` should follow the [kernel signature](kernel_signature.md), in the order:
1. All input tensors (may also contain scalar values as configurations)
2. All output tensors

`forward` returns nothing. The output tensors are allcated in advance and passed in as arguments
to avoid the overhead of allocating them in the function.

The scalar input can take in torch.Tensor with shape `[]`, or a simple Python scalar.

The kernel signature should try to align with [FlashInfer's API](https://docs.flashinfer.ai/).

## Example

```python
import torch

@torch.compile
def rmsnorm(x, eps=1e-5):
    return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

def forward(x, eps=1e-5):
    return rmsnorm(x, eps)
```
