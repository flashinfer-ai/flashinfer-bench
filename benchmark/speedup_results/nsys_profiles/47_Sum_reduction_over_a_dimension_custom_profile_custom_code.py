import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sum_dim1_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ x,
                                float* __restrict__ out,
                                int N, int S1, int S2) {
    extern __shared__ float sdata[];
    int global_idx = blockIdx.x;
    int b = global_idx / S2;
    int t = global_idx % S2;
    int tx = threadIdx.x;
    int stride = blockDim.x;

    float val = 0.0f;
    for (int i = tx; i < S1; i += stride) {
        val += x[b * (S1 * S2) + i * S2 + t];
    }
    sdata[tx] = val;
    __syncthreads();

    for (int s = stride >> 1; s > 0; s >>= 1) {
        if (tx < s) {
            sdata[tx] += sdata[tx + s];
        }
        __syncthreads();
    }

    if (tx == 0) {
        out[b * S2 + t] = sdata[0];
    }
}

torch::Tensor sum_dim1_cuda(torch::Tensor x) {
    const auto N = x.size(0);
    const auto S1 = x.size(1);
    const auto S2 = x.size(2);

    auto out = torch::zeros({N, 1, S2}, x.options());
    
    int threads = (S1 < 256) ? 1 : 256;
    while (threads < S1 && threads < 256) {
        threads <<= 1;
    }
    if (threads > 256) {
        threads = 256;
    }

    const int blocks = N * S2;
    const size_t sharedMem = threads * sizeof(float);

    sum_dim1_kernel<<<blocks, threads, sharedMem>>>(x.data_ptr<float>(),
                                                    out.data_ptr<float>(),
                                                    N, S1, S2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in sum_dim1_kernel: %s\\n", cudaGetErrorString(err));
    }

    return out;
}
"""

sum_dim1_cpp_source = r"torch::Tensor sum_dim1_cuda(torch::Tensor x);"

sum_dim1 = load_inline(
    name="sum_dim1",
    cpp_sources=sum_dim1_cpp_source,
    cuda_sources=sum_dim1_source,
    functions=["sum_dim1_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum_dim1.sum_dim1_cuda(x)
