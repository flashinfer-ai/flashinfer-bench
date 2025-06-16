import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_matmul_code = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void batched_matmul_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int batch_size, int M, int K, int N) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int batch = blockIdx.z;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int tiledRow = row;
        int tiledCol = t * BLOCK_SIZE + threadIdx.x;

        if (tiledRow < M && tiledCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[batch * (M * K) + tiledRow * K + tiledCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        tiledRow = t * BLOCK_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < K && tiledCol < N) {
            tileB[threadIdx.y][threadIdx.x] = B[batch * (K * N) + tiledRow * N + tiledCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        if (row < M && col < N) {
            for (int i = 0; i < BLOCK_SIZE; i++) {
                value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[batch * (M * N) + row * N + col] = value;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3, "A must be a 3D tensor");
    TORCH_CHECK(B.dim() == 3, "B must be a 3D tensor");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    TORCH_CHECK(B.size(0) == batch_size, "Batch size must match");
    TORCH_CHECK(B.size(1) == K, "Shapes must be compatible (K dimension)");
    int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, A.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
              batch_size);

    batched_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        M,
        K,
        N
    );

    return C;
}
"""

batch_matmul_cpp_source = """
torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

batch_matmul = load_inline(
    name="batch_matmul",
    cpp_sources=batch_matmul_cpp_source,
    cuda_sources=batch_matmul_code,
    functions=["batched_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return batch_matmul.batched_matmul_cuda(A, B)

batch_size = 128
m = 128
k = 256
n = 512

def get_inputs():
    A = torch.randn(batch_size, m, k)
    B = torch.randn(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []
