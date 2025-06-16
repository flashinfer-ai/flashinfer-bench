import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// A straightforward depthwise 2D convolution kernel implementation.
//
// Each thread is responsible for computing exactly one element in the output
// tensor at index (n, c, h_out, w_out). We then sum over the kernel region
// for that channel c. The bias is optionally added (if not null).
//
// Dimensions:
//   input:  (N, C, H, W)
//   weight: (C, 1, kH, kW)  (for depthwise conv, groups = C)
//   bias:   (C)             (optional)
//
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C,
    const int H, const int W,
    const int kH, const int kW,
    const int stride, const int padding,
    const bool use_bias,
    const int H_out, const int W_out
) {
    // Global index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Total number of output elements
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // Decompose idx into (n, c, h_out, w_out)
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c = tmp % C;
    int n = tmp / C;

    // Compute the "center" of the kernel in input coords
    // for top-left corner of the filter for (h_out, w_out).
    int h_in_center = h_out * stride - padding;
    int w_in_center = w_out * stride - padding;

    float val = 0.0f;
    // Accumulate over the kernel region for channel c
    for (int ky = 0; ky < kH; ky++) {
        for (int kx = 0; kx < kW; kx++) {
            int h_in = h_in_center + ky;
            int w_in = w_in_center + kx;

            // Check boundaries
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                int w_idx  = (c * kH + ky) * kW + kx;   // Flattened index into weight
                val += input[in_idx] * weight[w_idx];
            }
        }
    }

    // Add bias if requested
    if (use_bias) {
        val += bias[c];
    }

    int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = val;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) {
      TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dim() == 4, "input must have 4 dimensions: (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must have 4 dimensions: (C, 1, kH, kW)");

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto kH = weight.size(2);
    auto kW = weight.size(3);

    // Calculate output height/width
    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;

    // Allocate output tensor
    auto out_options = input.options();
    auto output = torch::zeros({N, C, H_out, W_out}, out_options);

    // Flatten pointers
    const float* input_ptr  = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr   = nullptr;
    bool use_bias = false;
    if (bias.defined() && bias.numel() == C) {
        bias_ptr = bias.data_ptr<float>();
        use_bias = true;
    }

    float* output_ptr = output.data_ptr<float>();

    // Launch kernel
    int total = N * C * H_out * W_out;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    depthwise_conv2d_kernel<<<gridSize, blockSize>>>(
        input_ptr, weight_ptr, bias_ptr,
        output_ptr,
        N, C, H, W, kH, kW,
        stride, padding,
        use_bias,
        H_out, W_out
    );

    // Synchronize to catch any kernel launch errors
    cudaDeviceSynchronize();

    return output;
}
""";

depthwise_conv2d_cpp_source = r"torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with a custom CUDA kernel.
    Mirrors the signature of the original Model but replaces the operator with a custom kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weight/bias similarly to nn.Conv2d(in_channels, in_channels, groups=in_channels)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform depthwise convolution using the custom CUDA kernel.
        """
        if self.bias is None:
            b = torch.Tensor().cuda()  # empty tensor to indicate no bias
        else:
            b = self.bias
        return depthwise_conv2d.depthwise_conv2d_cuda(x, self.weight, b, self.stride, self.padding)
