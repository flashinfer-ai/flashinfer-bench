import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source for custom 1D average pooling
avg_pool_1d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// 1D Average Pooling kernel
__global__ void avg_pool_1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int L_in, int L_out,
    int kernel_size, int stride, int padding)
{
    // Each block along y dimension corresponds to an (n, c) pair.
    int nc = blockIdx.y;
    int n = nc / C;
    int c = nc % C;

    // Thread along x dimension corresponds to an output index in L_out.
    int i_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_out >= L_out) return;

    // Compute start of the input window
    int start = i_out * stride - padding;

    // Accumulate sum over the pooling window
    float sum_val = 0.0f;
    int valid_count = 0;
    for (int k = 0; k < kernel_size; k++) {
        int i_in = start + k;
        if (i_in >= 0 && i_in < L_in) {
            sum_val += x[(n * C + c) * L_in + i_in];
            valid_count++;
        }
    }
    // Write averaged result to output
    y[(n * C + c) * L_out + i_out] = sum_val / (float)kernel_size;
}

// Host function to interface with PyTorch
torch::Tensor avg_pool_1d_cuda(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding)
{
    // Ensure we are on CUDA
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");

    auto sizes = x.sizes();
    const int N = sizes[0];
    const int C = sizes[1];
    const int L_in = sizes[2];
    const int L_out = (L_in + 2 * padding - kernel_size) / stride + 1;

    // Allocate output tensor
    auto out = torch::empty({N, C, L_out}, x.options());

    // Flatten tensors for easier indexing in the kernel
    auto x_flat = x.contiguous();
    auto out_flat = out.contiguous();

    dim3 block(256);
    dim3 grid((L_out + block.x - 1) / block.x, N * C);

    avg_pool_1d_kernel<<<grid, block>>>(
        x_flat.data_ptr<float>(),
        out_flat.data_ptr<float>(),
        N, C, L_in, L_out,
        kernel_size, stride, padding
    );

    return out_flat.view({N, C, L_out});
}
"""

# Corresponding C++ header declaration
avg_pool_1d_cpp_source = r"""
torch::Tensor avg_pool_1d_cuda(
    torch::Tensor x,
    int kernel_size,
    int stride,
    int padding);
"""

# Load/compile the custom kernel
avg_pool_1d_module = load_inline(
    name="avg_pool_1d_module",
    cpp_sources=avg_pool_1d_cpp_source,
    cuda_sources=avg_pool_1d_source,
    functions=["avg_pool_1d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized 1D Average Pooling model using a custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool_1d_module.avg_pool_1d_cuda(
            x,
            self.kernel_size,
            self.stride,
            self.padding
        )
