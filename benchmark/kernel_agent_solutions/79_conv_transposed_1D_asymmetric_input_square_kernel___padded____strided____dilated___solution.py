import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source for a custom transposed 1D convolution operator
conv1d_transpose_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel for transposed 1D convolution
__global__ void conv1d_transpose_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_length,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias,
    int out_length
) {
    // Each thread corresponds to one element in the output tensor
    // N = batch_size * out_channels * out_length
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_length;
    if (index >= total_elements) return;

    // Decode (n, oc, l_out) from the linear index
    int n = index / (out_channels * out_length);
    int remainder = index % (out_channels * out_length);
    int oc = remainder / out_length;
    int l_out = remainder % out_length;

    // Compute the convolution result
    float val = 0.0f;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            // Compute the corresponding input index
            int top_index = l_out + padding - k * dilation;
            if (top_index % stride == 0) {
                int x_in = top_index / stride;
                if (x_in >= 0 && x_in < in_length) {
                    // Weight index: [oc, ic, k]
                    int w_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                    // Input index: [n, ic, x_in]
                    int i_idx = n * (in_channels * in_length) + ic * in_length + x_in;
                    val += input[i_idx] * weight[w_idx];
                }
            }
        }
    }
    // Add bias if applicable
    if (has_bias) {
        val += bias[oc];
    }
    // Write to output
    int out_idx = n * (out_channels * out_length) + oc * out_length + l_out;
    output[out_idx] = val;
}

torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    // Shapes
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_length = input.size(2);

    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    // Compute output length for the transposed convolution
    // (No output_padding is applied here)
    int out_length = (in_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1;

    // Allocate output
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, out_length}, options);

    // Launch kernel
    int total_elements = batch_size * out_channels * out_length;
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;

    conv1d_transpose_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_length,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias,
        out_length
    );
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in conv1d_transpose_kernel: " + std::string(cudaGetErrorString(err)));
    }
    return output;
}
""";

conv1d_transpose_cpp_source = r"""
torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);
""";

# Build the custom operator
conv1d_transpose = load_inline(
    name="conv1d_transpose",
    cpp_sources=conv1d_transpose_cpp_source,
    cuda_sources=conv1d_transpose_cuda_source,
    functions=["conv1d_transpose_cuda"],
    extra_cflags=["-std=c++14"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Custom transposed 1D convolution with a fused custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        # Initialize weights and bias similarly to PyTorch
        weight = torch.empty(out_channels, in_channels, kernel_size)
        nn.init.kaiming_uniform_(weight, a=5**0.5)
        self.weight = nn.Parameter(weight)

        if bias:
            b = torch.zeros(out_channels)
            self.bias = nn.Parameter(b)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_bias = (self.bias is not None)
        if not has_bias:
            # If no bias is present, create a placeholder to pass to the kernel
            bias_tensor = torch.zeros(1, device=x.device, dtype=x.dtype)
        else:
            bias_tensor = self.bias

        # Call our custom CUDA operator
        return conv1d_transpose.conv1d_transpose_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            bias_tensor.contiguous(),
            self.stride,
            self.padding,
            self.dilation,
            has_bias
        )
