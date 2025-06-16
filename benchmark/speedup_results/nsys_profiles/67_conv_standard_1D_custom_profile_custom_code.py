import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Naive 1D convolution kernel with optional bias (groups=1 only)
__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, // batch size
    int C_in, // in_channels
    int C_out, // out_channels
    int in_width,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    // Indexing scheme: each thread handles one output element
    // blockIdx.x * blockDim.x + threadIdx.x -> flatten (N, C_out, out_width)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elems = N * C_out * out_width;
    if (idx >= total_output_elems) {
        return;
    }

    // Decompose idx to (n, co, w_out)
    int n = idx / (C_out * out_width);
    int r = idx % (C_out * out_width);
    int co = r / out_width;
    int w_out = r % out_width;

    float val = 0.0f;
    // Compute convolution
    for (int ci = 0; ci < C_in; ci++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_pos = w_out * stride - padding + k * dilation;
            if (in_pos >= 0 && in_pos < in_width) {
                int input_index = n * (C_in * in_width) + ci * in_width + in_pos;
                int weight_index = co * (C_in * kernel_size) + ci * kernel_size + k;
                val += input[input_index] * weight[weight_index];
            }
        }
    }
    if (has_bias) {
        val += bias[co];
    }
    output[idx] = val;
}

// C++ interface to call the CUDA kernel
torch::Tensor conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    // input shape = (N, C_in, in_width)
    // weight shape = (C_out, C_in, kernel_size)
    // bias shape = (C_out,) if has_bias
    int N = input.size(0);
    int C_in = input.size(1);
    int in_width = input.size(2);

    int C_out = weight.size(0);
    int kernel_size = weight.size(2);

    // Compute output width
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({N, C_out, out_width}, options);

    int total_output_elems = N * C_out * out_width;
    const int block_size = 256;
    int grid_size = (total_output_elems + block_size - 1) / block_size;

    conv1d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N,
        C_in,
        C_out,
        in_width,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias
    );

    return output;
}
""".replace('\n', '\n')  # Ensure newlines are preserved

conv1d_cpp_source = """
torch::Tensor conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);
"""

# Compile the CUDA code
conv1d_extension = load_inline(
    name="conv1d_extension",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_source,
    functions=["conv1d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized 1D Convolution with a custom CUDA kernel (groups=1 only).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super(ModelNew, self).__init__()
        if groups != 1:
            raise ValueError("Custom kernel only supports groups=1.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

        # Register weight and bias as parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv1d_extension.conv1d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.tensor([]).to(x.device),
            self.stride,
            self.padding,
            self.dilation,
            self.bias_flag
        )
