from pathlib import Path

import pytest
import torch

from flashinfer_bench.compile.builders import TileLangBuilder
from flashinfer_bench.data import Definition, Solution


@pytest.fixture
def gemm_definition() -> Definition:
    definition_json = {
        "name": "gemm_n128_k2048",
        "description": "General matrix multiply (GEMM) C = A @ B.T.",
        "op_type": "gemm",
        "tags": ["status:verified"],
        "axes": {
            "M": {"type": "var"},
            "N": {"type": "const", "value": 128},
            "K": {"type": "const", "value": 2048},
        },
        "inputs": {
            "A": {"shape": ["M", "K"], "dtype": "float16"},
            "B": {"shape": ["N", "K"], "dtype": "float16"},
        },
        "outputs": {"C": {"shape": ["M", "N"], "dtype": "float16"}},
        "reference": "import torch\n\ndef run(A, B):\n    C = torch.matmul(A, B.T)\n    return C",
    }
    return Definition.model_validate(definition_json)


@pytest.fixture
def tilelang_gemm_solution() -> Solution:
    """TileLang GEMM solution for testing. Adapted from TileLang's GEMM example.
    https://github.com/tile-ai/tilelang/tree/main/examples"""

    kernel_code = """import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul_kernel(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def run(A, B):
    M, K = A.shape
    N, _ = B.shape

    kernel = matmul_kernel(M, N, K, 128, 128, 64, T.float16)
    return kernel(A, B)
"""

    solution_json = {
        "name": "gemm_n128_k2048_tilelang_v1",
        "definition": "gemm_n128_k2048",
        "description": "TileLang GEMM implementation optimized for N=128, K=2048",
        "author": "test",
        "spec": {
            "language": "tilelang",
            "target_hardware": ["cuda"],
            "entry_point": "main.py::run",
            "dependencies": ["tilelang", "torch"],
            "destination_passing_style": False,
        },
        "sources": [{"path": "main.py", "content": kernel_code}],
    }
    return Solution.model_validate(solution_json)


@pytest.mark.requires_torch_cuda
def test_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name, package=None):
        if name == "tilelang":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    assert not TileLangBuilder.is_available()


@pytest.mark.requires_torch_cuda
@pytest.mark.skipif(not TileLangBuilder.is_available(), reason="tilelang not installed")
def test_tilelang_gemm_build_and_run(
    tmp_cache_dir: Path, gemm_definition: Definition, tilelang_gemm_solution: Solution
) -> None:
    builder = TileLangBuilder()

    assert builder.can_build(tilelang_gemm_solution)
    runnable = builder.build(gemm_definition, tilelang_gemm_solution)

    assert runnable.metadata.build_type == "tilelang"
    assert runnable.metadata.solution_name == "gemm_n128_k2048_tilelang_v1"
    assert runnable.metadata.destination_passing_style is False

    M = 256
    N = 128
    K = 2048

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")

    C = runnable(A, B)
    C_ref = torch.matmul(A, B.T)

    diff = torch.abs(C - C_ref).max()
    print(f"Max difference: {diff}")
    assert diff < 1e-2, f"Result differs from reference by {diff}"


@pytest.mark.requires_torch_cuda
@pytest.mark.skipif(not TileLangBuilder.is_available(), reason="tilelang not installed")
def test_tilelang_gemm_variable_m(
    tmp_cache_dir: Path, gemm_definition: Definition, tilelang_gemm_solution: Solution
) -> None:
    builder = TileLangBuilder()
    runnable = builder.build(gemm_definition, tilelang_gemm_solution)

    for M in [64, 128, 512, 1024]:
        N = 128
        K = 2048

        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(N, K, dtype=torch.float16, device="cuda")

        C = runnable(A, B)

        C_ref = torch.matmul(A, B.T)
        diff = torch.abs(C - C_ref).max()
        assert diff < 1e-2, f"Failed for M={M}, diff={diff}"
