import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv3d_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel to perform the scatter-based 3D transposed convolution accumulation.
// input shape:  (B, inC, inD, inH, inW)
// weight shape: (inC, outC_per_group, kD, kH, kW)   [outC = outC_per_group * groups]
// bias shape:   (outC)  [optional, only if bias is true]
// output shape: (B, outC, outD, outH, outW)
// stride, padding, output_padding, groups are given.
// This kernel uses atomicAdd to accumulate into the output tensor.
//
// index mapping for each (b, ic, id, ih, iw) thread:
//   1) Determine group g = ic // (inC / groups).
//   2) For each oc in [g*outC_per_group, (g+1)*outC_per_group):
//       for kd in [0..kD-1], kh in [0..kH-1], kw in [0..kW-1]:
//         outD = id * strideD - padD + kd
//         outH = ih * strideH - padH + kh
//         outW = iw * strideW - padW + kw
//         if 0 <= outD < OD and 0 <= outH < OH and 0 <= outW < OW:
//           atomicAdd(&output[b, oc, outD, outH, outW],
//                     input[b, ic, id, ih, iw] * weight[ic, oc - g*outC_per_group, kd, kh, kw]);
//
// After accumulation, if bias is defined, we launch another kernel to add bias[oc].
//
// For simplicity, we assume dilation = 1 here.

__global__ void transposed_conv3d_forward_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int B, const int inC, const int inD, const int inH, const int inW,
    const int outC, const int kD, const int kH, const int kW,
    const int outD, const int outH, const int outW,
    const int strideD, const int strideH, const int strideW,
    const int padD, const int padH, const int padW,
    const int outputPadD, const int outputPadH, const int outputPadW,
    const int groups)
{
    // Each thread handles a single (b, ic, id, ih, iw)
    // Compute global index:
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_in = B * inC * inD * inH * inW;
    if (idx >= total_in) return;

    // Decompose idx into (b, ic, id, ih, iw):
    int temp = idx;
    int iw_ = temp % inW;
    temp /= inW;
    int ih_ = temp % inH;
    temp /= inH;
    int id_ = temp % inD;
    temp /= inD;
    int ic_ = temp % inC;
    temp /= inC;
    int b_  = temp;

    // group logic
    int inC_per_group = inC / groups;
    int outC_per_group = outC / groups;
    int g_ = ic_ / inC_per_group; // which group

    // Now, read the input value once
    float in_val = input[idx];

    // offset pointers in weight accordingly
    // weight shape = [inC, outC_per_group, kD, kH, kW]
    const float* weight_ptr = weight + (ic_ * outC_per_group * kD * kH * kW);

    // Scatter to output
    for (int ocg = 0; ocg < outC_per_group; ocg++) {
        int oc_index = g_ * outC_per_group + ocg;
        for (int kd = 0; kd < kD; kd++) {
            int odepth = id_ * strideD - padD + kd;
            if (odepth < 0 || odepth >= outD) continue;

            for (int kh = 0; kh < kH; kh++) {
                int oheight = ih_ * strideH - padH + kh;
                if (oheight < 0 || oheight >= outH) continue;

                for (int kw_ = 0; kw_ < kW; kw_++) {
                    int owidth = iw_ * strideW - padW + kw_;
                    if (owidth < 0 || owidth >= outW) continue;

                    // weight offset
                    float w_val = weight_ptr[ocg * (kD * kH * kW)
                                             + kd * (kH * kW)
                                             + kh * kW
                                             + kw_];

                    // output offset
                    size_t out_idx = ((size_t)b_ * outC * outD * outH * outW)
                                   + ((size_t)oc_index * outD * outH * outW)
                                   + ((size_t)odepth * outH * outW)
                                   + ((size_t)oheight * outW)
                                   + owidth;

                    atomicAdd(&output[out_idx], in_val * w_val);
                }
            }
        }
    }
}

// Kernel to add bias to the output tensor
__global__ void add_bias_kernel(
    float* output,
    const float* bias,
    int B, int outC, int outD, int outH, int outW)
{
    // Each thread handles a single element in the output
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * outC * outD * outH * outW;
    if (idx >= total_out) return;

    int temp = idx;
    int w_ = temp % outW;
    temp /= outW;
    int h_ = temp % outH;
    temp /= outH;
    int d_ = temp % outD;
    temp /= outD;
    int oc_ = temp % outC;
    temp /= outC;
    // b_ = temp; // not needed beyond correctness check

    output[idx] += bias[oc_];
}

torch::Tensor transposed_conv3d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int groups,
    bool has_bias)
{
    // input shape:  B x inC x inD x inH x inW
    auto B = input.size(0);
    auto inC = input.size(1);
    auto inD = input.size(2);
    auto inH = input.size(3);
    auto inW = input.size(4);

    // weight shape: inC x (outC_per_group) x kD x kH x kW
    auto kD = weight.size(2);
    auto kH = weight.size(3);
    auto kW = weight.size(4);

    // outC = outC_per_group * groups
    auto outC = weight.size(1) * groups;

    // compute output dimensions (assuming dilation=1)
    int64_t strideD = stride[0], strideH = stride[1], strideW = stride[2];
    int64_t padD    = padding[0], padH    = padding[1], padW    = padding[2];
    int64_t outPadD = output_padding[0], outPadH = output_padding[1], outPadW = output_padding[2];

    // outD = (inD - 1)*strideD - 2*padD + kD + outPadD
    // outH = (inH - 1)*strideH - 2*padH + kH + outPadH
    // outW = (inW - 1)*strideW - 2*padW + kW + outPadW
    int64_t outD = (inD - 1) * strideD - 2 * padD + kD + outPadD;
    int64_t outH = (inH - 1) * strideH - 2 * padH + kH + outPadH;
    int64_t outW = (inW - 1) * strideW - 2 * padW + kW + outPadW;

    // create output
    auto out_options = input.options().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({B, outC, outD, outH, outW}, out_options);

    // launch kernel
    int threads = 256;
    int total_in = B * inC * inD * inH * inW;
    int blocks = (total_in + threads - 1) / threads;

    transposed_conv3d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, inC, inD, inH, inW,
        outC,
        kD, kH, kW,
        outD, outH, outW,
        strideD, strideH, strideW,
        padD, padH, padW,
        outPadD, outPadH, outPadW,
        groups
    );

    if (has_bias) {
      // Add bias
      int total_out = B * outC * outD * outH * outW;
      int bias_blocks = (total_out + threads - 1) / threads;
      add_bias_kernel<<<bias_blocks, threads>>>(
          output.data_ptr<float>(),
          bias.data_ptr<float>(),
          B, outC, outD, outH, outW
      );
    }
    return output;
}
"""

transposed_conv3d_cpp_source = r"""
torch::Tensor transposed_conv3d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int groups,
    bool has_bias
);
"""

# Compile the inline CUDA code
transposed_conv3d = load_inline(
    name="transposed_conv3d",
    cpp_sources=transposed_conv3d_cpp_source,
    cuda_sources=transposed_conv3d_source,
    functions=["transposed_conv3d_cuda_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Custom 3D transposed convolution with a scatter-based CUDA implementation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False
    ):
        super(ModelNew, self).__init__()
        # Register parameters similarly to nn.ConvTranspose3d
        # weight shape: (in_channels, out_channels // groups, kD, kH, kW)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        kD, kH, kW = kernel_size
        out_c_per_group = out_channels // groups

        # Weight
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_c_per_group, kD, kH, kW)
        )
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform custom transposed convolution via our CUDA kernel.
        """
        if self.bias is not None:
            return transposed_conv3d.transposed_conv3d_cuda_forward(
                x, self.weight, self.bias,
                list(self.stride), list(self.padding), list(self.output_padding),
                self.groups, True
            )
        else:
            dummy_bias = torch.empty(0, device=x.device, dtype=x.dtype)
            return transposed_conv3d.transposed_conv3d_cuda_forward(
                x, self.weight, dummy_bias,
                list(self.stride), list(self.padding), list(self.output_padding),
                self.groups, False
            )
