import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_mm_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_mm_kernel(const float* A, const float* B, float* C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        int idx = row * M + col;
        C[idx] = A[row] * B[idx];
    }
}

torch::Tensor diag_mm_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor for the diagonal");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Size mismatch between A and B dimensions");

    int N = A.size(0);
    int M = B.size(1);

    auto C = torch::empty_like(B);

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    diag_mm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "diag_mm_kernel launch failed with error: ", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "diag_mm_kernel failed to synchronize: ", cudaGetErrorString(err));

    return C;
}
"""

diag_mm_cpp_source = "torch::Tensor diag_mm_cuda(torch::Tensor A, torch::Tensor B);"

diag_mm = load_inline(
    name="diag_mm",
    cpp_sources=diag_mm_cpp_source,
    cuda_sources=diag_mm_source,
    functions=["diag_mm_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix using a custom CUDA kernel.
    C = diag(A) * B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_mm = diag_mm

    def forward(self, A, B):
        return self.diag_mm.diag_mm_cuda(A, B)

M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []
