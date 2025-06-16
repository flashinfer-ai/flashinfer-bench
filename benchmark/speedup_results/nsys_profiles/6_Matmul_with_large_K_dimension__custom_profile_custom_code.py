import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_tiled_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    for (int tile_idx = 0; tile_idx < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile_idx++) {
        int tiledCol = tile_idx * BLOCK_SIZE + threadIdx.x;
        int tiledRow = tile_idx * BLOCK_SIZE + threadIdx.y;

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

        for (int k = 0; k < BLOCK_SIZE; k++) {
            val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    auto out = torch::zeros({M, N}, A.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N
    );

    return out;
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
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_tiled = matmul_tiled

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_tiled.matmul_tiled_cuda(A, B)

M = 256
N = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
