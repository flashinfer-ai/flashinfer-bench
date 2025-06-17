"""
The implementation of SiluAndMul using flashinfer and native PyTorch.

Dtype: bfloat16
Shapes:
- x: (B, 2 * intermediate_size)
- out: (B, intermediate_size)
"""

import os
import flashinfer
import torch
import torch.nn.functional as F


def forward_flashinfer(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    flashinfer.activation.silu_and_mul(x, out)
    return out

def forward_pytorch(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

def kernel_descriptions():
    return """
silu_and_mul(x)

Shape variables:
- B: symbolic shape, batch_size
- intermediate_size: static constant, per-model

Input shapes:
- x: (B, 2 * intermediate_size)
- output: (B, intermediate_size)

Computation:
- output = silu(x[..., :intermediate_size]) * x[..., intermediate_size:]
"""

if __name__ == "__main__":
    # Check with `nvidia-smi --query-gpu=compute_cap --format=csv`
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
    # Constants
    B = 16
    D = 14336  # intermediate_size of Llama3.1-8B
    DTYPE = torch.bfloat16
    DEVICE = "cuda"

    # Create inputs
    x = torch.randn(B, D, dtype=DTYPE, device=DEVICE)

    # Test
    print("silu_and_mul")
    out_native = forward_pytorch(x)
    out_flashinfer = forward_flashinfer(x)
    torch.testing.assert_close(out_native, out_flashinfer)
