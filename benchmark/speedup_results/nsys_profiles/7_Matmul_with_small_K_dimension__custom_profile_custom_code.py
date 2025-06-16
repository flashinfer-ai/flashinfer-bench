import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tiled_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Tile dimensions
#ifndef TILE_DIM
#define TILE_DIM 16
#endif

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    float val = 0.0f;
    // Loop over tiles in K dimension
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int tiledCol = t * TILE_DIM + threadIdx.x;
        int tiledRow = t * TILE_DIM + threadIdx.y;

        // Load data into shared memory
        if (row < M && tiledCol < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tiledRow < K) {
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        // Compute the product for this tile
        for (int i = 0; i < TILE_DIM; i++) {
            val += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::dtype(A.dtype()).device(A.device()));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_tiled_kernel<<<grid, block>>>(A.data_ptr<float>(), 
                                         B.data_ptr<float>(), 
                                         C.data_ptr<float>(), 
                                         M, K, N);

    return C;
}
"""

matmul_tiled_cpp_source = r"torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B);"

matmul_tiled = load_inline(
    name="matmul_tiled",
    cpp_sources=matmul_tiled_cpp_source,
    cuda_sources=matmul_tiled_source,
    functions=["matmul_tiled_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using tiled shared memory in CUDA.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tiled_fn = matmul_tiled.matmul_tiled_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_tiled_fn(A, B)
