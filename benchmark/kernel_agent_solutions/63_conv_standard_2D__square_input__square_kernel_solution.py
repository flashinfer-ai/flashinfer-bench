import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Combined C++/CUDA source for a naive 2D convolution forward pass.
# This implementation does not support groups > 1 for simplicity.
# It demonstrates a straightforward, unoptimized approach that can be
# further enhanced with techniques like shared memory usage, loop unrolling,
# and better memory access patterns.
conv2d_forward_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Naive CUDA kernel for 2D convolution forward pass (no groups).
// Each thread computes a single output element in (N, outC, outH, outW).
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,         // batch size
    const int inC,       // input channels
    const int inH,       // input height
    const int inW,       // input width
    const int outC,      // output channels
    const int outH,      // output height
    const int outW,      // output width
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const bool use_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * outC * outH * outW;
    if (idx >= total_outputs) {
        return;
    }

    // Decode idx into (n, oc, oh, ow)
    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int oc = (idx / (outW * outH)) % outC;
    int n =  idx / (outW * outH * outC);

    // Compute starting point in the input
    // oh_out = oh, mapped to oh_in = oh * stride - padding + (kh * dilation)
    // ow_out = ow, mapped to ow_in = ow * stride - padding + (kw * dilation)
    float val = 0.0f;
    for (int ic = 0; ic < inC; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    int input_idx = n * inC * inH * inW
                                  + ic * inH * inW
                                  + ih * inW
                                  + iw;
                    int weight_idx = oc * inC * kernel_size * kernel_size
                                   + ic * kernel_size * kernel_size
                                   + kh * kernel_size
                                   + kw;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    if (use_bias) {
        val += bias[oc];
    }

    int out_idx = n * outC * outH * outW
                + oc * outH * outW
                + oh * outW
                + ow;
    output[out_idx] = val;
}

// Helper to compute output size for given conv params.
std::vector<int64_t> conv_output_size(
    int64_t inH,
    int64_t inW,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    int64_t outH = (inH + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t outW = (inW + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    return {outH, outW};
}

// Forward pass interface for PyTorch
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation
) {
    // input shape: (N, inC, inH, inW)
    // weight shape: (outC, inC, kernel_size, kernel_size)
    // bias shape: (outC) or empty if no bias

    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor.");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D tensor.");

    const int64_t N = input.size(0);
    const int64_t inC = input.size(1);
    const int64_t inH = input.size(2);
    const int64_t inW = input.size(3);
    const int64_t outC = weight.size(0);
    const int64_t kernel_size = weight.size(2);

    bool use_bias = (bias.numel() != 0);

    // Compute output size
    auto out_sizes = conv_output_size(inH, inW, kernel_size, stride, padding, dilation);
    const int outH = out_sizes[0];
    const int outW = out_sizes[1];

    // Create output tensor
    auto options = input.options();
    auto output = torch::zeros({N, outC, outH, outW}, options);

    // Launch kernel
    const int threads = 256;
    const int total_outputs = N * outC * outH * outW;
    const int blocks = (total_outputs + threads - 1) / threads;

    conv2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        use_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, inC, inH, inW, outC, outH, outW,
        kernel_size, stride, padding, dilation,
        use_bias
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_forward, "Naive Conv2D forward (CUDA)");
}
"""

# Build/Load the inline extension
my_custom_conv2d = load_inline(
    name="my_custom_conv2d",
    cpp_sources="",
    cuda_sources=conv2d_forward_source,
    functions=["conv2d_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Custom model that replaces the PyTorch Conv2D operator with a naive CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super().__init__()
        # For simplicity, we do not implement 'groups' here. 
        # Initialize parameters just like nn.Conv2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights/bias similar to PyTorch default for Conv2d
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that uses the custom CUDA kernel for 2D convolution.
        """
        if self.bias is None:
            b = torch.zeros(0, device=x.device, dtype=x.dtype)
        else:
            b = self.bias
        return my_custom_conv2d.conv2d_forward(
            x, self.weight, b,
            self.stride, self.padding, self.dilation
        )
