import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA code for ReLU
relu_kernel_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

torch::Tensor relu_forward(torch::Tensor input) {
    // Ensure the tensor is contiguous
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    int size = input_contig.numel();
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size>>>(input_contig.data_ptr<float>(),
                                           output.data_ptr<float>(),
                                           size);

    return output;
}
"""

# C++ function declaration
relu_kernel_cpp = r"""
torch::Tensor relu_forward(torch::Tensor input);
"""

# Load and compile the custom ReLU kernel
relu_module = load_inline(
    name="relu_module",
    cpp_sources=relu_kernel_cpp,
    cuda_sources=relu_kernel_source,
    functions=["relu_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation followed by a custom CUDA-based ReLU.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose3d(x)
        x = relu_module.relu_forward(x)
        return x
