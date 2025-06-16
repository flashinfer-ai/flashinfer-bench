import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_upper_tri_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float val = 0.0f;

    // Loop over tiles along the K dimension
    for(int t = 0; t < N / BLOCK_SIZE; t++) {
        int Acol = t * BLOCK_SIZE + threadIdx.x;
        int Brow = t * BLOCK_SIZE + threadIdx.y;

        // Load tile from A into shared memory (upper triangular condition)
        if (row < N && Acol < N && row <= Acol) {
            As[threadIdx.y][threadIdx.x] = A[row * N + Acol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory (upper triangular condition)
        if (Brow < N && col < N && Brow <= col) {
            Bs[threadIdx.y][threadIdx.x] = B[Brow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the loaded tiles
        for (int k = 0; k < BLOCK_SIZE; k++) {
            val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Store only the upper triangular result
    if (row < N && col < N) {
        if (col >= row) {
            C[row * N + col] = val;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

torch::Tensor matmul_upper_triangular_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    upper_triangular_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
"""

matmul_upper_tri_decl = r"""
torch::Tensor matmul_upper_triangular_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_upper_tri = load_inline(
    name="matmul_upper_tri",
    cpp_sources=matmul_upper_tri_decl,
    cuda_sources=matmul_upper_tri_source,
    functions=["matmul_upper_triangular_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) for upper triangular matrices using tiled shared-memory CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using a custom CUDA kernel.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return matmul_upper_tri.matmul_upper_triangular_cuda(A, B)

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []
