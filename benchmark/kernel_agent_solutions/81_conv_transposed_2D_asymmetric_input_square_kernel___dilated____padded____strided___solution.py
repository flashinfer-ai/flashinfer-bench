import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv2d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel for 2D transposed convolution
__global__ void transposed_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int stride, int padding, int dilation,
    bool hasBias
) {
    // Linear thread index over the output dimensions (N, outC, outH, outW)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * outC * outH * outW;
    if (index >= total_outputs) return;

    // Decompose linear index into (n, cOut, hOut, wOut)
    int wOut = index % outW;
    int tmp = index / outW;
    int hOut = tmp % outH;
    tmp = tmp / outH;
    int cOut = tmp % outC;
    int n = tmp / outC;

    // Accumulate result
    float val = 0.0f;
    // Iterate over input channels, kernel height, kernel width
    for(int cIn = 0; cIn < inC; cIn++) {
        for(int kh = 0; kh < kernelSize; kh++) {
            // Compute corresponding input h index
            int h_in_ = hOut + padding - kh * dilation;
            if(h_in_ % stride == 0) {
                int hIn = h_in_ / stride;
                if(hIn >= 0 && hIn < inH) {
                    for(int kw = 0; kw < kernelSize; kw++) {
                        // Compute corresponding input w index
                        int w_in_ = wOut + padding - kw * dilation;
                        if(w_in_ % stride == 0) {
                            int wIn = w_in_ / stride;
                            if(wIn >= 0 && wIn < inW) {
                                float inp = input[n * (inC * inH * inW)
                                               + cIn * (inH * inW)
                                               + hIn * inW
                                               + wIn];
                                float wval = weight[cIn * (outC * kernelSize * kernelSize)
                                                  + cOut * (kernelSize * kernelSize)
                                                  + kh * kernelSize
                                                  + kw];
                                val += inp * wval;
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias if present
    if (hasBias) {
        val += bias[cOut];
    }

    // Write output
    output[n * (outC * outH * outW)
           + cOut * (outH * outW)
           + hOut * outW
           + wOut] = val;
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
) {
    // Shapes and sizes
    const auto N = input.size(0);
    const auto inC = input.size(1);
    const auto inH = input.size(2);
    const auto inW = input.size(3);

    const auto outC = weight.size(1);
    const auto kernelSize = weight.size(2);

    // Compute output height/width based on transposed convolution formula
    const int outH = (inH - 1) * stride - 2 * padding + (kernelSize - 1) * dilation + 1;
    const int outW = (inW - 1) * stride - 2 * padding + (kernelSize - 1) * dilation + 1;

    // Prepare output tensor
    auto options = input.options();
    auto output = torch::zeros({N, outC, outH, outW}, options);

    bool hasBias = (bias.numel() > 0);
    // Configure CUDA kernel launch
    int total_outputs = N * outC * outH * outW;
    const int block_size = 256;
    const int grid_size = (total_outputs + block_size - 1) / block_size;

    transposed_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, inC, inH, inW,
        outC, outH, outW,
        kernelSize, stride, padding, dilation,
        hasBias
    );
    return output;
}
""";

transposed_conv2d_cpp_source = r"""
torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
);
""";

transposed_conv2d = load_inline(
    name="transposed_conv2d",
    cpp_sources=transposed_conv2d_cpp_source,
    cuda_sources=transposed_conv2d_source,
    functions=["transposed_conv2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Custom transposed convolution with CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        # Weight shape for ConvTranspose2d: (inC, outC, kH, kW)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            bias_tensor = torch.zeros((0,), device=x.device, dtype=x.dtype)
        else:
            bias_tensor = self.bias
        return transposed_conv2d.transposed_conv2d_cuda(
            x, self.weight, bias_tensor,
            self.stride, self.padding, self.dilation
        )
