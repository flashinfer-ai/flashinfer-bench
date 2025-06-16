import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# C++ function signature
# ------------------------------------------------------------
max_pool1d_cpp_source = r"""
torch::Tensor max_pool1d_forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation);
"""

# ------------------------------------------------------------
# CUDA source code (kernel + function definition)
# ------------------------------------------------------------
max_pool1d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for forward pass of Max Pool1D
__global__ void max_pool1d_forward_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          const int batch_size,
                                          const int channels,
                                          const int in_length,
                                          const int out_length,
                                          const int kernel_size,
                                          const int stride,
                                          const int padding,
                                          const int dilation) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batch_size * channels * out_length;
    if (idx >= total) return;

    // Compute (n, c, t_out) from the flattened index
    int n = idx / (channels * out_length);
    int remainder = idx % (channels * out_length);
    int c = remainder / out_length;
    int t_out = remainder % out_length;

    int start = t_out * stride - padding;
    float max_val = -3.402823466e+38F; // -FLT_MAX

    // Traverse the kernel window to find maximum
    for (int k = 0; k < kernel_size; k++) {
        int in_idx = start + k * dilation;
        if (in_idx >= 0 && in_idx < in_length) {
            float val = input[n * channels * in_length + c * in_length + in_idx];
            max_val = (val > max_val) ? val : max_val;
        }
    }
    output[idx] = max_val;
}

// C++ interface, called from Python
torch::Tensor max_pool1d_forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Extract input shapes
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_length = input.size(2);

    // Calculate output length similar to PyTorch's formula
    int out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output = torch::empty({batch_size, channels, out_length}, input.options());

    int total = batch_size * channels * out_length;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    // Launch CUDA kernel
    max_pool1d_forward_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_length,
        out_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    return output;
}
"""

# ------------------------------------------------------------
# Build the custom extension
# ------------------------------------------------------------
max_pool1d_opt = load_inline(
    name="max_pool1d_opt",
    cpp_sources=[max_pool1d_cpp_source],
    cuda_sources=[max_pool1d_source],
    functions=["max_pool1d_forward_cuda"],
    verbose=True
)

# ------------------------------------------------------------
# New model definition using the custom CUDA operator
# ------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        # return_indices is a PyTorch MaxPool1D feature; not implemented here
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return max_pool1d_opt.max_pool1d_forward_cuda(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )
