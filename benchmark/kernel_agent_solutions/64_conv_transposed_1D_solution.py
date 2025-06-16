import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv1d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void transposed_conv1d_forward_kernel(
    const float* __restrict__ input,   // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K]
    const float* __restrict__ bias,    // [C_out] or empty if no bias
    float* __restrict__ output,        // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding, bool has_bias)
{
    // Each block covers an (n, c_out, out_x) triple
    int out_x = blockIdx.x;
    int c_out = blockIdx.y;
    int n = blockIdx.z;

    if (n >= N || c_out >= C_out || out_x >= L_out) {
        return;
    }

    float val = 0.0f;
    // Accumulate over all in_channels and kernel positions
    for (int c_in_i = 0; c_in_i < C_in; c_in_i++) {
        for (int k_i = 0; k_i < K; k_i++) {
            int x_in = out_x + padding - k_i;
            // Check if it matches a valid input index after considering stride
            if (x_in % stride == 0) {
                x_in /= stride;
                if (x_in >= 0 && x_in < L_in) {
                    // input offset: n*C_in*L_in + c_in_i*L_in + x_in
                    // weight offset: c_in_i*C_out*K + c_out*K + k_i
                    float in_val = input[n * C_in * L_in + c_in_i * L_in + x_in];
                    float w_val = weight[c_in_i * C_out * K + c_out * K + k_i];
                    val += in_val * w_val;
                }
            }
        }
    }
    if (has_bias) {
        val += bias[c_out];
    }
    // output offset: n*C_out*L_out + c_out*L_out + out_x
    output[n * C_out * L_out + c_out * L_out + out_x] = val;
}

torch::Tensor transposed_conv1d_forward_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    bool has_bias
) {
    // input: [N, C_in, L_in]
    // weight: [C_in, C_out, K]
    // bias: [C_out] or empty
    // Output shape:
    //   L_out = (L_in - 1)*stride - 2*padding + K + output_padding
    int N = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);

    int C_out = weight.size(1);
    int K = weight.size(2);

    int L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto output = torch::zeros({N, C_out, L_out}, options);

    const int threads_x = 1;
    const int threads_y = 1;
    const int threads_z = 1;
    dim3 blockDim(threads_x, threads_y, threads_z);

    dim3 gridDim(L_out, C_out, N);

    transposed_conv1d_forward_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out,
        K, stride, padding, output_padding, has_bias
    );
    return output;
}
""";

transposed_conv1d_cpp_source = r"""
torch::Tensor transposed_conv1d_forward_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    bool has_bias
);
""";

transposed_conv1d = load_inline(
    name="transposed_conv1d",
    cpp_sources=transposed_conv1d_cpp_source,
    cuda_sources=transposed_conv1d_source,
    functions=["transposed_conv1d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Custom Transposed 1D Convolution using a CUDA kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()
        # For simplicity, we ignore groups beyond 1 in the custom kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias

        # Match PyTorch weight shape for ConvTranspose1d: [in_channels, out_channels, kernel_size]
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights (simple init)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the custom transposed 1D convolution.

        Args:
            x (torch.Tensor): Input of shape (N, C_in, L_in).

        Returns:
            torch.Tensor: Output of shape (N, C_out, L_out).
        """
        if self.bias_flag:
            return transposed_conv1d.transposed_conv1d_forward_cuda(
                x, self.weight, self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                True
            )
        else:
            # Pass an empty tensor for bias
            empty_bias = torch.empty(0, device=x.device, dtype=x.dtype)
            return transposed_conv1d.transposed_conv1d_forward_cuda(
                x, self.weight, empty_bias,
                self.stride,
                self.padding,
                self.output_padding,
                False
            )
