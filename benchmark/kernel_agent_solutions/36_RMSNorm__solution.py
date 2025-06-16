import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rms_norm_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Utility kernel to perform RMS Norm in one pass across features.
// Input/Output Layout: NCHW
// The normalization is computed over the C dimension only.
//
// x:  [N, F, H, W]
// out: [N, F, H, W]
// eps: float
//
// Steps per (n, h, w):
// 1) Accumulate sum of squares over all features (f) -> sumVal
// 2) Compute rms = sqrt( sumVal / F + eps )
// 3) Divide each x by rms and store in out
//
__global__ void rms_norm_forward_kernel(const float* __restrict__ x,
                                        float* __restrict__ out,
                                        const int N,
                                        const int F,
                                        const int H,
                                        const int W,
                                        const float eps) {
    // Calculate spatial coordinates
    int b = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= N || h >= H || w >= W) {
        return;
    }

    // Compute the squared sum over the feature dimension
    float sumVal = 0.0f;
    for (int f = 0; f < F; f++) {
        int idx = ((b * F + f) * H + h) * W + w;
        float val = x[idx];
        sumVal += val * val;
    }
    float meanVal = sumVal / (float)F;
    float rms = sqrtf(meanVal + eps);
    float invRms = 1.0f / rms;

    // Normalize across features
    for (int f = 0; f < F; f++) {
        int idx = ((b * F + f) * H + h) * W + w;
        out[idx] = x[idx] * invRms;
    }
}

torch::Tensor rms_norm_forward_cuda(torch::Tensor x, float eps) {
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (NCHW).");
    int N = x.size(0);
    int F = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty_like(x);

    // Configure block and grid sizes
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y,
              N);

    rms_norm_forward_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, F, H, W,
        eps
    );

    return out;
}
""";

// Function declaration for the inline C++ extension
std::vector<torch::Tensor> rms_norm_forward_cuda(
    torch::Tensor x,
    float eps
);
"""

rms_norm_cpp_source = r"""
std::vector<torch::Tensor> rms_norm_forward_cuda(
    torch::Tensor x,
    float eps
);
"""

# Compile the inline CUDA code
rms_norm_module = load_inline(
    name="rms_norm_module",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_forward_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using a custom CUDA kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self._rms_norm_module = rms_norm_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._rms_norm_module.rms_norm_forward_cuda(x, self.eps)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2, device="cuda")
    return [x]

def get_init_inputs():
    return [features]
