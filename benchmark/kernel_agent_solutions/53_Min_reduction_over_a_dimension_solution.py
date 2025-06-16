import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduce_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_reduce_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  int N, int D, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * M) {
        int n = idx / M;
        int m = idx % M;
        float current_min = x[n * D * M + m];
        for (int d = 1; d < D; d++) {
            float val = x[n * D * M + d * M + m];
            if (val < current_min) {
                current_min = val;
            }
        }
        out[n * M + m] = current_min;
    }
}

torch::Tensor min_reduce_cuda(torch::Tensor x, int dim) {
    TORCH_CHECK(x.dim() == 3, "Only supports 3D tensor.");
    TORCH_CHECK(dim == 1, "Only supports reduction along dimension=1.");

    int N = x.size(0);
    int D = x.size(1);
    int M = x.size(2);

    auto out = torch::zeros({N, M}, x.options());
    int blockSize = 256;
    int gridSize = (N * M + blockSize - 1) / blockSize;

    min_reduce_kernel<<<gridSize, blockSize>>>(x.data_ptr<float>(),
                                               out.data_ptr<float>(),
                                               N, D, M);
    return out;
}
"""

min_reduce_decl = "torch::Tensor min_reduce_cuda(torch::Tensor x, int dim);"

min_reduce = load_inline(
    name="min_reduce",
    cpp_sources=min_reduce_decl,
    cuda_sources=min_reduce_source,
    functions=["min_reduce_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return min_reduce.min_reduce_cuda(x, self.dim)
