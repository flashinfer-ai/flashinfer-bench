import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

groupnorm_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel to compute mean and variance per group using shared memory reduction
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var,
    int B, int C, int H, int W, int G)
{
    extern __shared__ float sdata[];
    float* smean = sdata;
    float* svar  = sdata + blockDim.x;

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index >= B * G) {
        return;
    }

    int b = index / G;
    int g = index % G;
    int group_size = C / G;
    int c_start = g * group_size;
    int c_end   = c_start + group_size;
    int count   = group_size * H * W;

    float sum   = 0.0f;
    float sqsum = 0.0f;

    for (int c = c_start; c < c_end; c++) {
        int offset = b * C * H * W + c * H * W;
        for (int i = 0; i < H * W; i++) {
            float val = static_cast<float>(input[offset + i]);
            sum += val;
            sqsum += val * val;
        }
    }

    smean[tid] = sum;
    svar[tid]  = sqsum;
    __syncthreads();

    // In-block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smean[tid] += smean[tid + stride];
            svar[tid]  += svar[tid + stride];
        }
        __syncthreads();
    }

    // Write final block values to global memory
    if (tid == 0) {
        float finalMean = smean[0] / count;
        float finalVar  = svar[0]  / count - finalMean * finalMean;
        mean[blockIdx.x] = finalMean;
        var[blockIdx.x]  = finalVar;
    }
}

// Kernel to apply normalization
template <typename scalar_t>
__global__ void apply_kernel(
    scalar_t* __restrict__ input,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const float eps,
    int B, int C, int H, int W, int G)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H * W) {
        return;
    }

    int n  = idx / (C * H * W);
    int cHW = idx % (C * H * W);
    int c  = cHW / (H * W);

    int group_size = C / G;
    int g = c / group_size;

    float m = mean[n * G + g];
    float v = var[n * G + g];
    float val = static_cast<float>(input[idx]);

    val = (val - m) / sqrtf(v + eps);
    input[idx] = static_cast<scalar_t>(val);
}

// Main forward function that computes mean/var and applies normalization
torch::Tensor groupnorm_forward(torch::Tensor x, int num_groups, float eps) {
    TORCH_CHECK(x.dim() == 4, "Input must be 4D: (N, C, H, W)");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA device.");

    auto B = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    auto mean = torch::empty({B * num_groups}, x.options().dtype(torch::kFloat));
    auto var  = torch::empty({B * num_groups}, x.options().dtype(torch::kFloat));

    const int threads = 256;
    const int blocks_stats = (B * num_groups + threads - 1) / threads;

    compute_stats_kernel<float><<<blocks_stats, threads, threads * 2 * sizeof(float)>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        B, C, H, W, num_groups
    );

    const int total_elems = B * C * H * W;
    const int blocks_apply = (total_elems + threads - 1) / threads;

    apply_kernel<float><<<blocks_apply, threads>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        eps,
        B, C, H, W, num_groups
    );

    return x;
}

// pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("groupnorm_forward",
          &groupnorm_forward,
          "GroupNorm forward (CUDA)");
}
"""

groupnorm_module = load_inline(
    name="groupnorm_module",
    cpp_sources=groupnorm_source,
    functions=["groupnorm_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Model that performs Group Normalization using a custom CUDA kernel.
    Supports affine transformation (weight and bias).
    """
    def __init__(self, num_features: int, num_groups: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = groupnorm_module.groupnorm_forward(x, self.num_groups, self.eps)
        if self.weight is not None and self.bias is not None:
            N, C, H, W = out.shape
            out = out * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return out
