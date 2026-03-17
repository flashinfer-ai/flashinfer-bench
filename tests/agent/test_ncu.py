"""Integration tests for NCU profiling agent API."""

from pathlib import Path

import pytest

from flashinfer_bench.agents.ncu import flashinfer_bench_run_ncu
from flashinfer_bench.data import Solution, TraceSet
from flashinfer_bench.data.workload import Workload

TRACE_SET_PATH = str(Path(__file__).resolve().parents[2] / "tmp-repos" / "flashinfer-trace")
DEFN_NAME = "gemm_n128_k2048"


def _load_gemm_fixture():
    ts = TraceSet.from_path(TRACE_SET_PATH)
    solution = ts.solutions[DEFN_NAME][0]
    workload = ts.workloads[DEFN_NAME][0].workload
    return solution, workload


@pytest.mark.requires_torch_cuda
def test_run_ncu_profiles_kernel():
    solution, workload = _load_gemm_fixture()
    result = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        trace_set_path=TRACE_SET_PATH,
        set="basic",
        page="details",
        timeout=120,
    )
    assert not result.startswith("ERROR:"), f"NCU returned error: {result}"
    assert "Section:" in result, f"No profiling sections in output: {result[:500]}"


@pytest.mark.requires_torch_cuda
def test_run_ncu_invalid_definition():
    solution, workload = _load_gemm_fixture()
    bad_solution = Solution(
        name="fake",
        definition="nonexistent_def_xyz",
        author="test",
        spec=solution.spec,
        sources=solution.sources,
    )
    result = flashinfer_bench_run_ncu(
        solution=bad_solution, workload=workload, trace_set_path=TRACE_SET_PATH
    )
    assert result.startswith("ERROR:")
    assert "not found" in result


@pytest.mark.requires_torch_cuda
def test_run_ncu_invalid_page():
    solution, workload = _load_gemm_fixture()
    result = flashinfer_bench_run_ncu(
        solution=solution, workload=workload, trace_set_path=TRACE_SET_PATH, page="foo"
    )
    assert result.startswith("ERROR:")
    assert "Invalid page" in result


def test_run_ncu_solution_file_not_found():
    workload = Workload(axes={}, inputs={}, uuid="test-stub")
    result = flashinfer_bench_run_ncu(
        solution="/nonexistent/path/solution.json", workload=workload, trace_set_path=TRACE_SET_PATH
    )
    assert result.startswith("ERROR:")
    assert "Solution file not found" in result


@pytest.mark.requires_torch_cuda
def test_run_ncu_timeout():
    solution, workload = _load_gemm_fixture()
    result = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        trace_set_path=TRACE_SET_PATH,
        set="basic",
        page="details",
        timeout=1,
    )
    assert result.startswith("ERROR:")
    assert "timed out" in result
