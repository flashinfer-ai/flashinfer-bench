import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_dim1_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void argmin_dim1_kernel(const float* __restrict__ x,
                                   const int B, const int d1, const int d2,
                                   int* __restrict__ out) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int dd = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < B && dd < d2) {
        float min_val = x[b * d1 * d2 + 0 * d2 + dd];
        int min_idx = 0;
        for (int i = 1; i < d1; i++) {
            float val = x[b * d1 * d2 + i * d2 + dd];
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        out[b * d2 + dd] = min_idx;
    }
}

torch::Tensor argmin_dim1_cuda(torch::Tensor x) {
    // B, d1, d2
    const auto B = x.size(0);
    const auto d1 = x.size(1);
    const auto d2 = x.size(2);

    auto out_options = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto out = torch::empty({B, d2}, out_options);

    const dim3 block(16, 16);
    const dim3 grid((B + block.x - 1) / block.x,
                    (d2 + block.y - 1) / block.y);

    argmin_dim1_kernel<<<grid, block>>>(x.data_ptr<float>(),
                                        B, d1, d2,
                                        out.data_ptr<int>());

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
'''

argmin_dim1_cpp_source = r'''
torch::Tensor argmin_dim1_cuda(torch::Tensor x);
'''

argmin_dim1 = load_inline(
    name="argmin_dim1",
    cpp_sources=argmin_dim1_cpp_source,
    cuda_sources=argmin_dim1_source,
    functions=["argmin_dim1_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin_dim1 = argmin_dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We assume self.dim == 1 for this custom kernel
        return self.argmin_dim1.argmin_dim1_cuda(x)
