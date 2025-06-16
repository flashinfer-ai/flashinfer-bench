import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv2d_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Utility macro for checking CUDA errors
#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void transposed_conv2d_kernel(
    const float* input,       // [N, IC, IH, IW]
    const float* weight,      // [IC, OC, KH, KW]
    const float* bias,        // [OC] or nullptr
    float* output,            // [N, OC, OH, OW]
    const int N,
    const int IC,
    const int IH,
    const int IW,
    const int OC,
    const int KH,
    const int KW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int OH,
    const int OW,
    const bool has_bias
) {
    // 3D grid decomposition:
    //   blockIdx.z => grouping over N*OC
    //   blockIdx.y => output height dimension
    //   blockIdx.x => output width dimension

    int zi = blockIdx.z;
    int n = zi / OC;         // batch index
    int oc = zi % OC;        // output channel index

    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (oh >= OH || ow >= OW) {
        return;
    }

    float val = 0.0f;
    // Accumulate over all input channels and kernel elements
    for (int ic = 0; ic < IC; ++ic) {
        for (int kh_i = 0; kh_i < KH; ++kh_i) {
            for (int kw_i = 0; kw_i < KW; ++kw_i) {
                int in_h = oh * strideH + kh_i - padH;
                int in_w = ow * strideW + kw_i - padW;
                // Check boundaries
                if (in_h >= 0 && in_h < IH && in_w >= 0 && in_w < IW) {
                    int input_idx = ((n * IC + ic) * IH + in_h) * IW + in_w;
                    int weight_idx = ((ic * OC + oc) * KH + kh_i) * KW + kw_i;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    // Add bias if present
    if (has_bias) {
        val += bias[oc];
    }
    // Store result
    int out_idx = ((n * OC + oc) * OH + oh) * OW + ow;
    output[out_idx] = val;
}

torch::Tensor transposed_conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideH,
    int strideW,
    int padH,
    int padW
) {
    // input shape: [N, IC, IH, IW]
    // weight shape: [IC, OC, KH, KW]
    // bias shape: [OC] or empty

    int N = input.size(0);
    int IC = input.size(1);
    int IH = input.size(2);
    int IW = input.size(3);

    int ICw = weight.size(0);  // should match input's IC
    int OC = weight.size(1);
    int KH = weight.size(2);
    int KW = weight.size(3);

    bool has_bias = bias.defined() && (bias.numel() == OC);

    // Compute output height/width for transposed conv
    // formula for transposed conv output:
    // OH = (IH - 1) * strideH - 2*padH + KH
    // OW = (IW - 1) * strideW - 2*padW + KW
    int OH = (IH - 1) * strideH - 2 * padH + KH;
    int OW = (IW - 1) * strideW - 2 * padW + KW;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({N, OC, OH, OW}, options);

    const int threads_x = 16;
    const int threads_y = 16;
    dim3 block(threads_x, threads_y);
    dim3 grid((OW + threads_x - 1) / threads_x,
              (OH + threads_y - 1) / threads_y,
              N * OC);

    transposed_conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N,
        IC,
        IH,
        IW,
        OC,
        KH,
        KW,
        strideH,
        strideW,
        padH,
        padW,
        OH,
        OW,
        has_bias
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return output;
}
""";

transposed_conv2d_cuda_hdr = r"""
torch::Tensor transposed_conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int strideH,
    int strideW,
    int padH,
    int padW
);
""";

_transposed_conv2d = load_inline(
    name="transposed_conv2d",
    cpp_sources=transposed_conv2d_cuda_hdr,
    cuda_sources=transposed_conv2d_cuda_src,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
    functions=["transposed_conv2d_cuda_forward"],
)

class ModelNew(nn.Module):
    """
    Custom transposed convolution using an inline CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        # Weight shape for nn.ConvTranspose2d in PyTorch: [in_channels, out_channels, kH, kW]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        kH, kW = kernel_size
        # Initialize weights and optional bias similarly to PyTorch's default initialization
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kH, kW))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strideH, strideW = self.stride
        padH, padW = self.padding
        bias_tensor = self.bias if self.has_bias else torch.tensor([], device=x.device)
        return _transposed_conv2d.transposed_conv2d_cuda_forward(
            x,
            self.weight,
            bias_tensor,
            strideH,
            strideW,
            padH,
            padW
        )
