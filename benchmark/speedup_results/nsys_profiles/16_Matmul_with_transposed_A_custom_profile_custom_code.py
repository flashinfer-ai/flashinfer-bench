import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tiled_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define TILE_SIZE 32

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    // 2D indices for this thread in block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 2D indices for this block in the output
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Shared memory for sub-tiles of A and B
    extern __shared__ float sharedMem[];
    float* As = sharedMem;
    float* Bs = sharedMem + TILE_SIZE * TILE_SIZE;

    float value = 0.0f;

    // Loop over tiles in k dimension
    for (int t = 0; t < K; t += TILE_SIZE) {
        // Load tile from A into shared memory:
        if ((row < M) && (t + tx < K)) {
            As[ty * TILE_SIZE + tx] = A[(t + tx) * M + row];
        } else {
            As[ty * TILE_SIZE + tx] = 0.0f;
        }

        // Load tile from B into shared memory:
        if ((col < N) && (t + ty < K)) {
            Bs[ty * TILE_SIZE + tx] = B[(t + ty) * N + col];
        } else {
            Bs[ty * TILE_SIZE + tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial products
        for (int i = 0; i < TILE_SIZE; i++) {
            value += As[ty * TILE_SIZE + i] * Bs[i * TILE_SIZE + tx];
        }
        __syncthreads();
    }

    // Store the result in C
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    // A is shape (K, M), B is shape (K, N)
    // We want output C of shape (M, N) = A.T x B
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);

    auto options = A.options();
    auto C = torch::zeros({M, N}, options);

    // Launch config
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    size_t sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    matmul_tiled_kernel<<<grid, block, sharedMemSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    // Error check after launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Kernel launch error: ") + cudaGetErrorString(err));
    }

    // Synchronize and check again
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Kernel sync error: ") + cudaGetErrorString(err));
    }

    return C;
}
"""

matmul_tiled_cpp_source = r"torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B);"

matmul_tiled = load_inline(
    name="matmul_tiled",
    cpp_sources=matmul_tiled_cpp_source,
    cuda_sources=matmul_tiled_source,
    functions=["matmul_tiled_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tiled = matmul_tiled

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compute C = A.T x B using the custom tiled kernel
        return self.matmul_tiled.matmul_tiled_cuda(A, B)
