import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tril_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

__global__ void tril_matmul_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float value = 0.0f;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int t = 0; t < n; t += BLOCK_SIZE) {
        int tiledCol = t + threadIdx.x;
        int tiledRow = t + threadIdx.y;

        // Load sub-matrix of A into shared memory
        if (row < n && tiledCol < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + tiledCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load sub-matrix of B into shared memory
        if (tiledRow < n && col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two matrices together
        for (int k = 0; k < BLOCK_SIZE; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to C, but only for the lower triangular portion
    if (row < n && col < n) {
        if (row >= col) {
            C[row * n + col] = value;
        } else {
            C[row * n + col] = 0.0f;
        }
    }
}

torch::Tensor matmul_lower_triangular_cuda(torch::Tensor A, torch::Tensor B) {
    int64_t n = A.size(0);
    auto C = torch::empty_like(A);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    tril_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        n
    );

    return C;
}
"""

matmul_tril_cpp_source = r"""
torch::Tensor matmul_lower_triangular_cuda(torch::Tensor A, torch::Tensor B);
"""

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) where A and B are lower triangular matrices, 
    using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tril = load_inline(
            name="matmul_tril",
            cpp_sources=matmul_tril_cpp_source,
            cuda_sources=matmul_tril_source,
            functions=["matmul_lower_triangular_cuda"],
            verbose=False
        )

    def forward(self, A, B):
        # Ensure the tensors are contiguous in memory for optimal performance
        return self.matmul_tril.matmul_lower_triangular_cuda(A.contiguous(), B.contiguous())

M = 4096

def get_inputs():
    A = torch.randn(M, M)
    B = torch.randn(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []
