import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tiled_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float val = 0.0f;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; m++) {
        int tiledRow = row;
        int tiledCol = m * TILE_SIZE + threadIdx.x;
        if (tiledRow < N && tiledCol < N) {
            tileA[threadIdx.y][threadIdx.x] = A[tiledRow * N + tiledCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        tiledRow = m * TILE_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < N && tiledCol < N) {
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            val += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    int64_t N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_kernel<<<gridSize, blockSize>>>(A.data_ptr<float>(),
                                                 B.data_ptr<float>(),
                                                 C.data_ptr<float>(), N);

    return C;
}
"""

matmul_cpp_source = r"torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B);"

matmul_tiled = load_inline(
    name="matmul_tiled",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_tiled_source,
    functions=["matmul_tiled_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tiled = matmul_tiled

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_tiled.matmul_tiled_cuda(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]

def get_init_inputs():
    return []
