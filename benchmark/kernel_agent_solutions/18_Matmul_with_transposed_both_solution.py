import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ code for a tiled matrix multiplication kernel
tiled_matmul_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int A_col = t * BLOCK_SIZE + threadIdx.x;
        int B_row = t * BLOCK_SIZE + threadIdx.y;

        if (row < M && A_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (B_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[B_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            val += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    // A is shape (M, K)
    // B is shape (K, N)
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor.");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor.");
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.size(0) == K, "B's first dimension must match A's second dimension.");
    int64_t N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_tiled_kernel<<<gridDim, blockDim>>>(A.data_ptr<float>(),
                                               B.data_ptr<float>(),
                                               C.data_ptr<float>(),
                                               M, K, N);

    return C;
}
"""

tiled_matmul_cpp_source = r"""
torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B);
"""

# Load the inline extension
tiled_matmul = load_inline(
    name="tiled_matmul",
    cpp_sources=tiled_matmul_cpp_source,
    cuda_sources=tiled_matmul_source,
    functions=["matmul_tiled_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tiled = tiled_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Transpose A and B to get shapes (M, K) and (K, N), then apply the custom kernel
        A_t = A.T.contiguous()
        B_t = B.T.contiguous()
        return self.matmul_tiled.matmul_tiled_cuda(A_t, B_t)

M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(K, M)
    B = torch.randn(N, K)
    return [A, B]

def get_init_inputs():
    return []
