import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// A tile size that is usually a decent starting point for 2D convolutions
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// Depthwise 2D convolution kernel
//   x:       [N, C, H_in, W_in]
//   w:       [C, K, K] (depthwise means out_channels == in_channels == C)
//   bias:    [C] (optional, used if has_bias=true)
//   out:     [N, C, H_out, W_out]
//   H_out = floor((H_in + 2*pad - K) / stride) + 1
//   W_out = floor((W_in + 2*pad - K) / stride) + 1
template <bool has_bias>
__global__ void depthwise_conv2d_kernel(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        const float* __restrict__ bias,
                                        float* __restrict__ out,
                                        const int N, const int C,
                                        const int H_in, const int W_in,
                                        const int K, const int stride,
                                        const int pad,
                                        const int H_out, const int W_out) {
    // 2D block coordinates
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute global output row/col from block + thread indices
    int out_col = bx * BLOCK_SIZE + tx;
    int out_row = by * BLOCK_SIZE + ty;

    // Loop over N, C
    // We rely on additional loops or a flattened approach
    // but to keep the kernel simpler, we just do the 2D region
    // for each thread and then loop over N*C inside.

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            // Check if the computed output pixel is within valid range
            if (out_row < H_out && out_col < W_out) {
                float value = 0.0f;

                // Compute each element
                for (int ky = 0; ky < K; ky++) {
                    for (int kx = 0; kx < K; kx++) {
                        int in_row = out_row * stride - pad + ky;
                        int in_col = out_col * stride - pad + kx;
                        // Check boundaries
                        if (in_row >= 0 && in_row < H_in &&
                            in_col >= 0 && in_col < W_in) {
                            // x index: n*C*H_in*W_in + c*H_in*W_in + in_row*W_in + in_col
                            float x_val = x[((n * C + c) * H_in + in_row) * W_in + in_col];
                            // w index: c*K*K + ky*K + kx
                            float w_val = w[(c * K + ky) * K + kx];
                            value += x_val * w_val;
                        }
                    }
                }
                if (has_bias) {
                    value += bias[c];
                }
                // out index: n*C*H_out*W_out + c*H_out*W_out + out_row*W_out + out_col
                out[((n * C + c) * H_out + out_row) * W_out + out_col] = value;
            }
        }
    }
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor x,
                                    torch::Tensor w,
                                    c10::optional<torch::Tensor> bias_opt,
                                    int stride,
                                    int pad) {
    // x shape: [N, C, H_in, W_in]
    // w shape: [C, K, K]
    // bias_opt shape: [C] if present

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(w.dim() == 3, "w must be 3D");
    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    TORCH_CHECK(w.size(0) == C, "weight shape[0] should match x.size(1)");
    int K = w.size(1);
    TORCH_CHECK(K == w.size(2), "Kernel must be square, w.size(1) == w.size(2)");

    int H_out = (H_in + 2*pad - K) / stride + 1;
    int W_out = (W_in + 2*pad - K) / stride + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "Output dimensions are non-positive");

    // Allocate output
    auto out = torch::zeros({N, C, H_out, W_out}, x.options());

    // Prepare block and grid
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (H_out + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.size(0) == C, "bias length must match channel count");
        depthwise_conv2d_kernel<true><<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C,
            H_in, W_in,
            K, stride, pad,
            H_out, W_out
        );
    } else {
        depthwise_conv2d_kernel<false><<<grid, block>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            nullptr,
            out.data_ptr<float>(),
            N, C,
            H_in, W_in,
            K, stride, pad,
            H_out, W_out
        );
    }

    return out;
}
""".strip()

depthwise_conv2d_cpp_decl = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor x,
                                    torch::Tensor w,
                                    c10::optional<torch::Tensor> bias_opt,
                                    int stride,
                                    int pad);
"""

# Load the CUDA extension
depthwise_conv2d_extension = load_inline(
    name="depthwise_conv2d_extension",
    cpp_sources=depthwise_conv2d_cpp_decl,
    cuda_sources=depthwise_conv2d_src,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Custom depthwise 2D convolution using a fused CUDA kernel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ):
        super().__init__()
        # We'll store weights/bias as separate tensors and run our custom kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Depthwise: out_channels == in_channels for group = in_channels
        # Create parameters
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We expect x shape [N, C, H, W].
        # Weight shape [C, kernel_size, kernel_size], bias shape [C] if present.
        return depthwise_conv2d_extension.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else None,
            self.stride,
            self.padding
        )
