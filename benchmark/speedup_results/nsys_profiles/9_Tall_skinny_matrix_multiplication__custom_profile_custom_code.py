import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tiled_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;
    // Loop over all tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tiledRow = row;
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        if (tiledRow < M && tiledCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[tiledRow * K + tiledCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        tiledRow = t * TILE_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < K && tiledCol < N) {
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    // Check device
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::runtime_error("Tensors must be on the same CUDA device");
    }

    // Dimensions
    int M = A.size(0);
    int K = A.size(1);
    int K2 = B.size(0);
    int N = B.size(1);

    // Ensure dimension match
    if (K != K2) {
        throw std::runtime_error("Inner dimensions must match for matmul");
    }

    // Create output tensor
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({M, N}, options);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                         B.data_ptr<float>(),
                                         C.data_ptr<float>(),
                                         M, K, N);

    // Optional: check for errors
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //    throw std::runtime_error(cudaGetErrorString(error));
    // }

    return C;
}
'''

matmul_tiled_cpp_source = r'''
torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B);
'''

matmul_tiled = load_inline(
    name="matmul_tiled",
    cpp_sources=matmul_tiled_cpp_source,
    cuda_sources=matmul_tiled_source,
    functions=["matmul_tiled_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tiled = matmul_tiled

    def forward(self, A, B):
        return self.matmul_tiled.matmul_tiled_cuda(A.contiguous(), B.contiguous())
