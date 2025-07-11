"""
The implementation of rmsnorm using flashinfer and native PyTorch.

Shapes:
- x: (B, H)
- weight: (H, )
- residual: (B, H)
"""

import os
import flashinfer
import torch
from typing import Optional, Union, Tuple


def forward_flashinfer(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if residual is not None:
        # NOTE: in-place operation
        flashinfer.norm.fused_add_rmsnorm(x, residual, weight, eps)
        return x, residual
    out = flashinfer.norm.rmsnorm(x, weight, eps=eps)
    return out

def forward_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Follows SGLang's https://github.com/sgl-project/sglang/blob/f20f70003d9546ec6e6d5b068b3c5f7ab0521109/python/sglang/srt/layers/layernorm.py
    """
    if not x.is_contiguous():
        x = x.contiguous()
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight).to(orig_dtype)
    if residual is None:
        return x
    else:
        return x, residual

def kernel_descriptions():
    return """
rmsnorm(x, weight, eps)

Shape Variables:
- B: symbolic shape, batch_size
- H: static constant, per-model

Input shapes:
- x: (B, H)
- weight: (H, )
- residual: (B, H)

Computation:
- Refer to forward_pytorch().
"""

if __name__ == "__main__":
    # Check with `nvidia-smi --query-gpu=compute_cap --format=csv`
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
    # Constants
    B = 16
    H = 4096
    EPS = 1e-5
    DTYPE = torch.bfloat16
    DEVICE = "cuda"

    # Create inputs
    weight = torch.randn(H, dtype=DTYPE, device=DEVICE)
    x = torch.randn(B, H, dtype=DTYPE, device=DEVICE)
    residual = torch.randn_like(x)

    # 1. rmsnom
    print("rmsnorm")
    out_native = forward_pytorch(x, weight, EPS)
    out_flashinfer = forward_flashinfer(x, weight, EPS)
    torch.testing.assert_close(out_native, out_flashinfer)

    # 2. fused_add_rmsnorm
    print("fused_add_rmsnorm")
    out_native = forward_pytorch(x, weight, EPS, residual)
    out_flashinfer = forward_flashinfer(x, weight, EPS, residual)
    torch.testing.assert_close(out_native, out_flashinfer)
