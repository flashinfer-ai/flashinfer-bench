import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tiled_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void matmul_kernel_tiled(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int tiledACol = t * BLOCK_SIZE + threadIdx.x;
        int tiledBRow = t * BLOCK_SIZE + threadIdx.y;

        // Load tile of A into shared memory
        if (row < M && tiledACol < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + tiledACol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory
        if (col < N && tiledBRow < K) {
            sB[threadIdx.y][threadIdx.x] = B[tiledBRow * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute partial products
        for (int i = 0; i < BLOCK_SIZE; i++) {
            val += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

    const int M = A.size(0);
    const int K = A.size(1);
    // B is expected to have shape (K, N)
    TORCH_CHECK(B.size(0) == K, "A's K dimension and B's K dimension must match");
    const int N = B.size(1);

    auto options = A.options();
    auto C = torch::zeros({M, N}, options);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel_tiled<<<grid, block>>>(A.data_ptr<float>(),
                                         B.data_ptr<float>(),
                                         C.data_ptr<float>(),
                                         M, K, N);

    return C;
}
"""

matmul_tiled_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for tiled matrix multiplication
matmul_tiled = load_inline(
    name="matmul_tiled",
    cpp_sources=matmul_tiled_cpp_source,
    cuda_sources=matmul_tiled_source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom tiled CUDA kernel for matrix multiplication.
    Expects:
        A of shape (M, K) on CUDA,
        B of shape (N, K) on CUDA,
        and performs torch.matmul(A, B.T) effectively.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tiled = matmul_tiled

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # We transpose B to get shape (K, N) for our kernel
        B_t = B.transpose(0, 1).contiguous()
        return self.matmul_tiled.matmul_cuda(A, B_t)
