import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_src = r'''
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

// Naive 3D convolution kernel (N, C_out, D_out, H_out, W_out) indexing via a 1D grid
__global__ void conv3d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int C_in,
    const int D_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int KD,
    const int KH,
    const int KW,
    const int D_out,
    const int H_out,
    const int W_out,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const bool use_bias
){
    // Total number of output elements
    int total_out = N * C_out * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    // Decode idx into (n, co, d_out, h_out, w_out)
    int w_out_idx = idx % W_out;
    int h_out_idx = (idx / W_out) % H_out;
    int d_out_idx = (idx / (W_out * H_out)) % D_out;
    int co = (idx / (W_out * H_out * D_out)) % C_out;
    int n  = idx / (W_out * H_out * D_out * C_out);

    // Compute starting point (input) in each dimension
    int d_in_origin = d_out_idx * stride_d - pad_d;
    int h_in_origin = h_out_idx * stride_h - pad_h;
    int w_in_origin = w_out_idx * stride_w - pad_w;

    float sum_val = 0.0f;
    // Convolution: sum over C_in, KD, KH, KW
    for(int ci = 0; ci < C_in; ci++){
        for(int kd = 0; kd < KD; kd++){
            for(int kh = 0; kh < KH; kh++){
                for(int kw_ = 0; kw_ < KW; kw_++){
                    int d_in = d_in_origin + kd * dilation_d;
                    int h_in = h_in_origin + kh * dilation_h;
                    int w_in = w_in_origin + kw_ * dilation_w;
                    // Check boundaries
                    if(d_in >= 0 && d_in < D_in &&
                       h_in >= 0 && h_in < H_in &&
                       w_in >= 0 && w_in < W_in){
                        int in_idx = ((n * C_in + ci) * D_in + d_in) * H_in * W_in
                                     + h_in * W_in + w_in;
                        int wt_idx = ((co * C_in + ci) * KD + kd) * KH * KW
                                     + kh * KW + kw_;
                        sum_val += input[in_idx] * weight[wt_idx];
                    }
                }
            }
        }
    }

    if(use_bias){
        sum_val += bias[co];
    }

    // Store result
    int out_idx = ((n * C_out + co) * D_out + d_out_idx) * H_out * W_out
                  + h_out_idx * W_out + w_out_idx;
    output[out_idx] = sum_val;
}

// Wrapper for Python
torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
){
    // We will assume groups = 1 for simplicity
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "input must be 5D (N, C_in, D, H, W)");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D (C_out, C_in, KD, KH, KW)");
    TORCH_CHECK(groups == 1, "Only groups=1 is supported in this sample kernel");

    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    int N = input_sizes[0];
    int C_in = input_sizes[1];
    int D_in = input_sizes[2];
    int H_in = input_sizes[3];
    int W_in = input_sizes[4];

    int C_out = weight_sizes[0];
    int K_cin = weight_sizes[1];
    int KD = weight_sizes[2];
    int KH = weight_sizes[3];
    int KW = weight_sizes[4];
    TORCH_CHECK(C_in == K_cin, "input channel must match weight channel");

    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    int dilation_d = dilation[0];
    int dilation_h = dilation[1];
    int dilation_w = dilation[2];

    // Calculate output dimensions
    int D_out = (D_in + 2*pad_d - dilation_d*(KD-1) - 1) / stride_d + 1;
    int H_out = (H_in + 2*pad_h - dilation_h*(KH-1) - 1) / stride_h + 1;
    int W_out = (W_in + 2*pad_w - dilation_w*(KW-1) - 1) / stride_w + 1;

    auto options = input.options();
    auto out = torch::zeros({N, C_out, D_out, H_out, W_out}, options);

    bool use_bias = false;
    const float* bias_ptr = nullptr;
    if(bias_opt.has_value()){
        auto bias_tensor = bias_opt.value();
        TORCH_CHECK(bias_tensor.dim() == 1, "bias must be 1D (C_out)");
        TORCH_CHECK(bias_tensor.size(0) == C_out, "bias length must match C_out");
        use_bias = true;
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    int total_out = N * C_out * D_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_out + block_size - 1) / block_size;

    conv3d_forward_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        out.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, KD, KH, KW,
        D_out, H_out, W_out,
        pad_d, pad_h, pad_w,
        stride_d, stride_h, stride_w,
        dilation_d, dilation_h, dilation_w,
        use_bias
    );

    return out;
}
'''

conv3d_cpp_header = r'''
torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
);
'''

# Compile the inline extension
conv3d_module = load_inline(
    name="conv3d_cuda_module",
    cpp_sources=conv3d_cpp_header,
    cuda_sources=conv3d_src,
    functions=["conv3d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Custom 3D convolution module with a naive CUDA kernel for demonstration.
    Only supports groups=1, bias optional.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        dilation: tuple = (1, 1, 1),
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()
        # Register parameters similarly to nn.Conv3d
        # kernel_size is a 3-tuple (KD, KH, KW)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # Weight shape: (out_channels, in_channels, KD, KH, KW)
        weight = torch.randn(
            out_channels,
            in_channels,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2]
        )
        self.weight = nn.Parameter(weight)
        if bias:
            b = torch.randn(out_channels)
            self.bias = nn.Parameter(b)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d_module.conv3d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else None,
            list(self.stride),
            list(self.padding),
            list(self.dilation),
            self.groups
        )
