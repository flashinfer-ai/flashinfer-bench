import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

matmul_tiled_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M,
                                    int K,
                                    int N) 
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float val = 0.0f;

    for (int m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; m++) {
        // Load A tile
        if (row < M && (m * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + (m * TILE_SIZE + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        if (col < N && (m * TILE_SIZE + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sums
        for (int i = 0; i < TILE_SIZE; i++) {
            val += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    const auto M = A.size(0);
    const auto K = A.size(1);
    const auto N = B.size(1);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({M, N}, options);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

matmul_tiled_cpp_source = r"torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B);"

matmul_tiled = load_inline(
    name="matmul_tiled",
    cpp_sources=matmul_tiled_cpp_source,
    cuda_sources=matmul_tiled_source,
    functions=["matmul_tiled_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tiled = matmul_tiled

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_tiled.matmul_tiled_cuda(A, B)
