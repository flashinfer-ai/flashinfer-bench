import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ code for optimized BatchNorm
batchnorm_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel to compute mean and variance for each channel using shared memory reduction
__global__ void compute_mean_var_kernel(const float* __restrict__ x,
                                        float* __restrict__ mean,
                                        float* __restrict__ var,
                                        int N, int C, int H, int W) {
    // N: batch size
    // C: channels
    // H, W: spatial dimensions
    // gridDim.x = C, each block processes one channel
    // blockDim.x = number of threads per block

    extern __shared__ float sdata[]; 
    float* ssum = sdata;                  // partial sums
    float* ssum_sq = sdata + blockDim.x;  // partial sums of squares

    int c = blockIdx.x; 
    int plane_size = N * H * W; 
    int tid = threadIdx.x;
    int offset = c * plane_size;

    // Initialize shared memory
    ssum[tid] = 0.0f;
    ssum_sq[tid] = 0.0f;
    __syncthreads();

    // Stride over the plane for the channel
    for(int idx = tid; idx < plane_size; idx += blockDim.x) {
        float val = x[offset + idx];
        ssum[tid]    += val;
        ssum_sq[tid] += val * val;
    }
    __syncthreads();

    // Reduction in shared memory
    // (assumes blockDim.x is a power of 2 for simplicity here)
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            ssum[tid]    += ssum[tid + stride];
            ssum_sq[tid] += ssum_sq[tid + stride];
        }
        __syncthreads();
    }

    // Write results to global mem (thread 0 only)
    if(tid == 0) {
        float m = ssum[0] / (float)plane_size;
        float msq = ssum_sq[0] / (float)plane_size;
        mean[c] = m;
        var[c]  = msq - m * m;
    }
}

// Kernel to apply batch normalization
__global__ void apply_batch_norm_kernel(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ var,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        const float eps,
                                        int N, int C, int H, int W) {
    // gridDim.x * gridDim.y * gridDim.z = total elements
    // each thread processes one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = N * C * H * W;
    if (idx >= size) return;

    // Calculate which channel this element belongs to
    int c = (idx / (H*W)) % C;
    float mean_val = mean[c];
    float var_val = var[c];
    float inv_std = rsqrtf(var_val + eps);

    float val = x[idx];
    // (val - mean) / sqrt(var + eps) * gamma + beta
    val = (val - mean_val) * inv_std;
    val = val * gamma[c] + beta[c];
    y[idx] = val;
}

// Host function that orchestrates the BN forward pass
torch::Tensor batch_norm_forward(torch::Tensor x,
                                 torch::Tensor gamma,
                                 torch::Tensor beta,
                                 float eps) {
    // x:    [N, C, H, W]
    // gamma,beta: [C]
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    auto mean = torch::zeros({C}, x.options());
    auto var  = torch::zeros({C}, x.options());
    auto y    = torch::empty_like(x);

    // Launch 1D grid of size C, each block for one channel
    const int threads_per_block = 256;
    compute_mean_var_kernel<<<C, threads_per_block, 2 * threads_per_block * sizeof(float)>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        N, C, H, W
    );

    // Launch apply kernel
    int total_size = N * C * H * W;
    int blocks = (total_size + threads_per_block - 1) / threads_per_block;
    apply_batch_norm_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        eps,
        N, C, H, W
    );

    return y;
}

TORCH_LIBRARY(my_optimized_bn, m) {
    m.def("batch_norm_forward", batch_norm_forward);
}
"""

batchnorm_cpp_decl = r"""
torch::Tensor batch_norm_forward(torch::Tensor x,
                                 torch::Tensor gamma,
                                 torch::Tensor beta,
                                 float eps);
"""

# Compile and load the inline extension
batchnorm_extension = load_inline(
    name="my_optimized_bn",
    cpp_sources=batchnorm_cpp_decl,
    cuda_sources=batchnorm_cuda_src,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    functions=["batch_norm_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Custom Batch Normalization model using an optimized CUDA kernel.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        # Parameters gamma, beta
        self.gamma = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))
        # For simplicity, we won't track running stats here (purely for demo).
        # If needed, you can add them and update in forward pass.

        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (N, C, H, W)
        # calls the custom BN
        return batchnorm_extension.batch_norm_forward(
            x, self.gamma, self.beta, self.eps
        )

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]
