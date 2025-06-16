import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

pointwise_conv2d_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void pointwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    bool use_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * height * width;
    if (idx >= total) return;

    // Decompose idx into b, oc, h, w
    int w_idx = idx % width;
    int h_idx = (idx / width) % height;
    int oc_idx = (idx / (width * height)) % out_channels;
    int b_idx = idx / (out_channels * height * width);

    float val = 0.0f;

    // Accumulate over in_channels
    for (int ic = 0; ic < in_channels; ic++) {
        // input index: (b_idx, ic, h_idx, w_idx)
        int input_idx = (((b_idx * in_channels) + ic) * height + h_idx) * width + w_idx;
        // weight index: (oc_idx, ic, 0, 0)
        int weight_idx = oc_idx * in_channels + ic;
        val += input[input_idx] * weight[weight_idx];
    }

    // Add bias if specified
    if (use_bias) {
        val += bias[oc_idx];
    }

    output[idx] = val;
}

torch::Tensor pointwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool use_bias)
{
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto out_channels = weight.size(0);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    int total = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    pointwise_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        use_bias
    );

    return output;
}
'''

pointwise_conv2d_cpp_source = """
torch::Tensor pointwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool use_bias);
"""

pointwise_conv2d = load_inline(
    name="pointwise_conv2d",
    cpp_sources=pointwise_conv2d_cpp_source,
    cuda_sources=pointwise_conv2d_source,
    functions=["pointwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Custom pointwise 2D convolution using a custom CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias

        # Initialize weights and bias similarly to PyTorch's nn.Conv2d for a 1x1 conv
        weight = torch.empty((out_channels, in_channels, 1, 1))
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight.view(out_channels, in_channels))

        if bias:
            bias_param = torch.empty(out_channels)
            fan_in = in_channels * 1 * 1
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias_param, -bound, bound)
            self.bias = nn.Parameter(bias_param)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            return pointwise_conv2d.pointwise_conv2d_cuda(
                x, self.weight, self.bias, True
            )
        else:
            # if no bias, we still pass a placeholder for bias
            dummy_bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
            return pointwise_conv2d.pointwise_conv2d_cuda(
                x, self.weight, dummy_bias, False
            )
