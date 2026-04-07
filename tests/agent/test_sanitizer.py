"""Integration tests for compute-sanitizer agent API."""

import pytest

from flashinfer_bench.agents.sanitizer import flashinfer_bench_run_sanitizer
from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages, TraceSet
from flashinfer_bench.env import get_fib_dataset_path

TRACE_SET_PATH = str(get_fib_dataset_path())
DEFN_NAME = "gemm_n128_k2048"


def _load_gemm_fixture():
    ts = TraceSet.from_path(TRACE_SET_PATH)
    solution = ts.solutions[DEFN_NAME][0]
    workload = ts.workloads[DEFN_NAME][0].workload
    return solution, workload


def _make_oob_solution():
    """Create a triton solution with intentional out-of-bounds memory access."""
    code = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def buggy_kernel(\n"
        "    A_ptr, B_ptr, C_ptr,\n"
        "    M, N: tl.constexpr, K: tl.constexpr,\n"
        "    stride_am, stride_ak,\n"
        "    stride_bn, stride_bk,\n"
        "    stride_cm, stride_cn,\n"
        "):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = tl.arange(0, N) + pid * N + 100000000\n"
        "    val = tl.load(A_ptr + offs)\n"
        "    tl.store(C_ptr + offs, val)\n"
        "\n"
        "def run(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):\n"
        "    M, K = A.shape\n"
        "    N = B.shape[0]\n"
        "    buggy_kernel[(M,)](\n"
        "        A, B, C, M, N, K,\n"
        "        A.stride(0), A.stride(1),\n"
        "        B.stride(0), B.stride(1),\n"
        "        C.stride(0), C.stride(1),\n"
        "    )\n"
    )
    return Solution(
        name="buggy_oob_kernel",
        definition=DEFN_NAME,
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=["NVIDIA_B200"],
            entry_point="main.py::run",
            destination_passing_style=True,
        ),
        sources=[SourceFile(path="main.py", content=code)],
    )


@pytest.mark.requires_torch_cuda
def test_run_sanitizer_all_tools_pass():
    solution, workload = _load_gemm_fixture()
    result = flashinfer_bench_run_sanitizer(
        solution=solution, workload=workload, trace_set_path=TRACE_SET_PATH, timeout=120
    )
    assert not result.startswith("ERROR:"), f"Sanitizer returned error: {result[:500]}"
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        assert f"{tool} passed successfully" in result, f"{tool} did not pass: {result[:1000]}"


@pytest.mark.requires_torch_cuda
def test_run_sanitizer_memcheck_detects_oob():
    _, workload = _load_gemm_fixture()
    buggy = _make_oob_solution()
    result = flashinfer_bench_run_sanitizer(
        solution=buggy,
        workload=workload,
        trace_set_path=TRACE_SET_PATH,
        sanitizer_types=["memcheck"],
        timeout=120,
    )
    assert "Invalid" in result, f"memcheck did not detect OOB: {result[:1000]}"
    assert "detected issues" in result


@pytest.mark.requires_torch_cuda
def test_run_sanitizer_invalid_definition():
    solution, workload = _load_gemm_fixture()
    bad_solution = Solution(
        name="fake",
        definition="nonexistent_def_xyz",
        author="test",
        spec=solution.spec,
        sources=solution.sources,
    )
    result = flashinfer_bench_run_sanitizer(
        solution=bad_solution, workload=workload, trace_set_path=TRACE_SET_PATH
    )
    assert result.startswith("ERROR:")
    assert "not found" in result
