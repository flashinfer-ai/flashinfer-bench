import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_optimized_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv1d_optimized_kernel(
    const float* __restrict__ input,     // [B, C_in, W_in]
    const float* __restrict__ weight,    // [C_out, C_in, K]
    const float* __restrict__ bias,      // [C_out] or empty if no bias
    float* __restrict__ output,          // [B, C_out, W_out]
    const int B, 
    const int C_in, 
    const int W_in,
    const int C_out, 
    const int K,
    const int stride,
    const int dilation,
    const int W_out,
    const bool has_bias)
{
    // 3D mapping of threads: (out_x, out_c, b)
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = blockIdx.y * blockDim.y + threadIdx.y;
    int b     = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_x < W_out && out_c < C_out && b < B) {
        float sum_val = 0.0f;
        // Compute convolution
        for (int in_c = 0; in_c < C_in; ++in_c) {
            for (int k = 0; k < K; ++k) {
                int in_x = out_x * stride + k * dilation;
                if (in_x >= 0 && in_x < W_in) {
                    // input index
                    int input_idx = b * C_in * W_in + in_c * W_in + in_x;
                    // weight index
                    int weight_idx = out_c * C_in * K + in_c * K + k;
                    sum_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
        if (has_bias) {
            sum_val += bias[out_c];
        }
        // write out
        int out_idx = b * C_out * W_out + out_c * W_out + out_x;
        output[out_idx] = sum_val;
    }
}

torch::Tensor conv1d_optimized_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation,
    bool has_bias)
{
    // input shape:  [B, C_in, W_in]
    // weight shape: [C_out, C_in, K]
    // bias shape:   [C_out] (if has_bias = true)
    // Output shape: [B, C_out, W_out]
    //   W_out = floor((W_in - dilation*(K-1) - 1) / stride + 1)

    int B      = input.size(0);
    int C_in   = input.size(1);
    int W_in   = input.size(2);
    int C_out  = weight.size(0);
    int K      = weight.size(2);
    int W_out  = ( (W_in - dilation * (K - 1) - 1 ) / stride ) + 1;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::zeros({B, C_out, W_out}, options);

    const int threads_x = 8;
    const int threads_y = 8;
    const int threads_z = 4;

    dim3 blockDim(threads_x, threads_y, threads_z);
    dim3 gridDim(
        (W_out + threads_x - 1) / threads_x,
        (C_out + threads_y - 1) / threads_y,
        (B + threads_z - 1) / threads_z);

    conv1d_optimized_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, C_in, W_in,
        C_out, K,
        stride, dilation,
        W_out,
        has_bias
    );

    return output;
}
''';

conv1d_optimized_cpp_source = r'''
torch::Tensor conv1d_optimized_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation,
    bool has_bias);
''';

conv1d_optimized_module = load_inline(
    name="conv1d_optimized_module",
    cpp_sources=conv1d_optimized_cpp_source,
    cuda_sources=conv1d_optimized_source,
    functions=["conv1d_optimized_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Custom 1D convolution that uses a fused CUDA kernel for computation.
    Mirrors torch.nn.Conv1d with stride, dilation, and optional bias.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias_flag = bias

        # Define weight, bias as learnable parameters
        weight_shape = (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is float and on the same device as weight
        x = x.to(self.weight.device, dtype=self.weight.dtype)
        has_bias = (self.bias is not None)
        if has_bias:
            return conv1d_optimized_module.conv1d_optimized_forward(
                x, self.weight, self.bias, self.stride, self.dilation, True
            )
        else:
            dummy_bias = torch.tensor([], device=x.device, dtype=x.dtype)
            return conv1d_optimized_module.conv1d_optimized_forward(
                x, self.weight, dummy_bias, self.stride, self.dilation, False
            )
