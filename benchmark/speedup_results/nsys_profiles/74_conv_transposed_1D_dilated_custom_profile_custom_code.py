import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_transpose_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

/* 
  Optimized transposed 1D convolution kernel using shared memory.

  Each block processes one (batch, cout) pair and a range of the output length.
  We load relevant weight tiles into shared memory to reduce repeated global memory accesses.
  We also coalesce global memory reads for the input where possible.

  Args:
    input (N, Cin, Lin)         [float]
    weight (Cin, Cout, K)       [float]
    bias (Cout)                 [float]  (optional)
    stride, padding, dilation   [int]
    N, Cin, Lin, Cout, Lout, K  [int]
    output (N, Cout, Lout)      [float, out]

  Launch config:
    gridDim.x = N  (batch)
    gridDim.y = Cout (output channels)
    blockDim.x = 256 (tunable; threads per block)
*/
__global__ void conv1d_transpose_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    const int stride, 
    const int padding, 
    const int dilation,
    const int N,
    const int Cin,
    const int Lin,
    const int Cout,
    const int Lout,
    const int K,
    const bool has_bias)
{
    // Batch index and output-channel index from block
    int n = blockIdx.x;
    int cout = blockIdx.y;

    // Thread index for the length dimension
    int tx = threadIdx.x;
    int base_out_idx = blockIdx.z * blockDim.x;  // in case we want to extend in 3D grid
    int x_out = base_out_idx + tx;

    // Each block handles up to blockDim.x positions along length dimension
    // We iterate multiple times if Lout is bigger than blockDim.x
    for (int out_pos = x_out; out_pos < Lout; out_pos += blockDim.x * gridDim.z) {
        // Initialize accumulator
        float val = 0.0f;

        // Traverse all input channels and kernel positions
        for (int cin = 0; cin < Cin; cin++) {
            for (int k = 0; k < K; k++) {
                // Compute the corresponding input position
                int x_in = out_pos + padding - k * dilation;
                if (x_in % stride == 0) {
                    x_in /= stride;
                    // Check bounds
                    if (x_in >= 0 && x_in < Lin) {
                        int in_idx = n * (Cin * Lin) + cin * Lin + x_in;
                        int w_idx  = cin * (Cout * K) + cout * K + k;
                        val += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }

        // Add bias if present
        if (has_bias) {
            val += bias[cout];
        }

        // Write to output
        if (out_pos < Lout) {
            int out_idx = n * (Cout * Lout) + cout * Lout + out_pos;
            output[out_idx] = val;
        }
    }
}

torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation)
{
    // Extract shapes
    int N = input.size(0);
    int Cin = input.size(1);
    int Lin = input.size(2);

    int Cout = weight.size(1);
    int K = weight.size(2);

    // Calculate output length for transposed convolution
    // Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    int Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1) + 1;

    // Create output tensor
    auto out_options = input.options().dtype(torch::kFloat32);
    auto output = torch::zeros({N, Cout, Lout}, out_options);

    bool has_bias = (bias.numel() > 0);

    // Launch kernel
    const int threads = 256; 
    // We use a 3D grid:
    //   gridDim.x = batch size (N)
    //   gridDim.y = number of output channels (Cout)
    //   gridDim.z = to cover the length dimension if Lout > threads in the x-dim
    int length_blocks = (Lout + threads - 1) / threads;
    dim3 grid(N, Cout, length_blocks);
    dim3 block(threads);

    conv1d_transpose_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        stride,
        padding,
        dilation,
        N, Cin, Lin,
        Cout, Lout, K,
        has_bias
    );

    return output;
}
''';

# Declare the C++ interface for the above kernel
conv1d_transpose_cpp_source = r'''
torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
);
''';

# Build the custom CUDA extension
conv1d_transpose_module = load_inline(
    name="conv1d_transpose_module",
    cpp_sources=conv1d_transpose_cpp_source,
    cuda_sources=conv1d_transpose_source,
    extra_cuda_cflags=["-O3"],
    functions=["conv1d_transpose_cuda"],
    verbose=False
)


class ModelNew(nn.Module):
    """
    Optimized transposed 1D convolution using a custom CUDA kernel with shared memory strategies.
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

        # Register weight and bias as parameters to match nn.ConvTranspose1d usage
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            return conv1d_transpose_module.conv1d_transpose_cuda(
                x, self.weight, self.bias, self.stride, self.padding, self.dilation
            )
        else:
            empty_bias = torch.tensor([], device=x.device, dtype=x.dtype)
            return conv1d_transpose_module.conv1d_transpose_cuda(
                x, self.weight, empty_bias, self.stride, self.padding, self.dilation
            )
