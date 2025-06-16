import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

mean_reduce_dim1_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel to reduce along dimension=1 for a tensor of shape [B, D1, D2].
// We accumulate partial sums in shared memory to improve memory access patterns.
__global__ void mean_reduce_dim1_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const int B,
                                        const int D1,
                                        const int D2)
{
    // b = batch index, col = column index
    int b   = blockIdx.x;
    int col = blockIdx.y;

    // Each block is responsible for one (b, col) pair.
    // We launch blockDim.x threads, each does partial sums across dimension=1.
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Initialize shared memory
    sdata[tid] = 0.0f;

    // Stride through dimension=1
    for (int row = tid; row < D1; row += blockDim.x) {
        sdata[tid] += input[b * D1 * D2 + row * D2 + col];
    }

    // Synchronize threads before reduction
    __syncthreads();

    // Perform a standard parallel reduction in shared memory
    // reduce blockDim.x elements
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write out the final sum from thread 0, then compute mean
    if (tid == 0) {
        output[b * D2 + col] = sdata[0] / (float)D1;
    }
}

// C++ function to call the kernel
torch::Tensor mean_reduce_dim1_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor [B, D1, D2].");
    auto B = input.size(0);
    auto D1 = input.size(1);
    auto D2 = input.size(2);

    // Allocate output of shape [B, D2]
    auto out_options = input.options().dtype(input.scalar_type());
    torch::Tensor output = torch::zeros({B, D2}, out_options);

    // Configure grid and block sizes
    // One block per (B, D2) pair, up to 256 threads
    dim3 grid(B, D2);
    int threads = (D1 < 256) ? D1 : 256;

    // Launch kernel with dynamic shared memory sized by 'threads'
    mean_reduce_dim1_kernel<<<grid, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        D1,
        D2
    );

    // Check for kernel launch errors
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_reduce_dim1_kernel launch failed with error:", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "cudaDeviceSynchronize failed with error:", cudaGetErrorString(err));

    return output;
}
"""

mean_reduce_dim1_cpp = r"torch::Tensor mean_reduce_dim1_cuda(torch::Tensor input);"

# Compile the inline CUDA code
mean_reduce_dim1 = load_inline(
    name="mean_reduce_dim1",
    cpp_sources=mean_reduce_dim1_cpp,
    cuda_sources=mean_reduce_dim1_source,
    functions=["mean_reduce_dim1_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over dimension=1 using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        super().__init__()
        # This code is specialized for dim=1
        # The provided dimension is stored for consistency, but only dim=1 is supported here.
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calls our custom kernel if dim=1, otherwise defaults to standard torch.mean.
        if self.dim == 1:
            return mean_reduce_dim1.mean_reduce_dim1_cuda(x)
        else:
            return torch.mean(x, dim=self.dim)
