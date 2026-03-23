"""Fixtures for serve API tests."""

import pytest
import pytest_asyncio
import torch

from flashinfer_bench.bench import BenchmarkConfig
from flashinfer_bench.data import (
    AxisConst,
    BuildSpec,
    Definition,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
)
from flashinfer_bench.serve.app import app, init_app
from flashinfer_bench.serve.scheduler import Scheduler

try:
    from httpx import ASGITransport, AsyncClient
except ImportError:
    pytest.skip("httpx not installed", allow_module_level=True)

pytest_plugins = ("pytest_asyncio",)


# ── Helpers ──


def _make_spec(entry_point: str = "pkg/main.py::kernel") -> BuildSpec:
    return BuildSpec(
        language=SupportedLanguages.PYTHON,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=False,
    )


def _make_solution(name: str, definition: str, code: str) -> Solution:
    return Solution(
        name=name,
        definition=definition,
        author="test",
        spec=_make_spec(),
        sources=[SourceFile(path="pkg/main.py", content=code)],
    )


# ── Test Definition ──


def make_test_definition() -> Definition:
    """Create a simple test definition: out = input * 2."""
    return Definition(
        name="test_scale",
        op_type="test",
        axes={"N": AxisConst(value=128)},
        inputs={"X": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"Y": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(X):\n    return X * 2\n",
    )


def make_test_workload() -> Workload:
    return Workload(axes={"N": 128}, inputs={"X": RandomInput()}, uuid="test_workload_001")


# ── Solution Fixtures ──


def solution_correct(definition: str) -> Solution:
    """A correct solution that passes all checks."""
    return _make_solution(
        "correct", definition, "import torch\n\ndef kernel(X):\n    return X * 2\n"
    )


def solution_syntax_error(definition: str) -> Solution:
    """A solution with syntax error (COMPILE_ERROR)."""
    return _make_solution(
        "syntax_error", definition, "def kernel(X\n    return X * 2\n"  # Missing closing paren
    )


def solution_runtime_crash(definition: str) -> Solution:
    """A solution that crashes at runtime with Python exception (non-fatal RUNTIME_ERROR)."""
    return _make_solution(
        "runtime_crash",
        definition,
        "import torch\n\ndef kernel(X):\n    raise ValueError('intentional crash')\n",
    )


def solution_wrong_shape(definition: str) -> Solution:
    """A solution that outputs wrong shape (INCORRECT_SHAPE)."""
    return _make_solution(
        "wrong_shape",
        definition,
        "import torch\n\ndef kernel(X):\n    return X[:64] * 2\n",  # Half the size
    )


def solution_wrong_value(definition: str) -> Solution:
    """A solution that outputs wrong values (INCORRECT_NUMERICAL)."""
    return _make_solution(
        "wrong_value",
        definition,
        "import torch\n\ndef kernel(X):\n    return X * 3\n",  # Wrong multiplier
    )


def solution_wrong_dtype(definition: str) -> Solution:
    """A solution that outputs wrong dtype (INCORRECT_DTYPE)."""
    return _make_solution(
        "wrong_dtype",
        definition,
        "import torch\n\ndef kernel(X):\n    return (X * 2).to(torch.float16)\n",
    )


def solution_slow(definition: str) -> Solution:
    """A solution that times out (TIMEOUT)."""
    return _make_solution(
        "slow",
        definition,
        "import torch\nimport time\n\ndef kernel(X):\n    time.sleep(100)\n    return X * 2\n",
    )


def solution_illegal_memory(definition: str) -> Solution:
    """A solution that causes illegal memory access (CUDA context corruption).

    Uses a Triton kernel that writes out of bounds to corrupt CUDA context.
    """
    code = """import torch
import triton
import triton.language as tl

@triton.jit
def corrupt_kernel(X_ptr, Y_ptr, N: tl.constexpr):
    # Write far out of bounds to cause illegal memory access
    offs = tl.arange(0, 128) + 1000000000  # Huge offset
    tl.store(Y_ptr + offs, tl.load(X_ptr + tl.arange(0, 128)))

def kernel(X):
    Y = torch.empty_like(X)
    corrupt_kernel[(1,)](X, Y, X.numel())
    torch.cuda.synchronize()  # Force error to surface
    return Y
"""
    return Solution(
        name="illegal_memory",
        definition=definition,
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=["cuda"],
            entry_point="pkg/main.py::kernel",
            destination_passing_style=False,
            dependencies=[],
        ),
        sources=[SourceFile(path="pkg/main.py", content=code)],
    )


# ── Pytest Fixtures ──


@pytest.fixture
def test_trace_set() -> TraceSet:
    """Create an in-memory TraceSet for testing."""
    definition = make_test_definition()
    workload = make_test_workload()

    trace_set = TraceSet()
    trace_set.definitions[definition.name] = definition
    trace_set.workloads[definition.name] = [
        Trace(definition=definition.name, workload=workload, solution=None, evaluation=None)
    ]
    return trace_set


@pytest.fixture
def benchmark_config() -> BenchmarkConfig:
    """Short timeout config for testing."""
    return BenchmarkConfig(warmup_runs=1, iterations=2, num_trials=1, timeout_seconds=5)


@pytest.fixture
def scheduler(test_trace_set, benchmark_config) -> Scheduler:
    """Create a real scheduler with one GPU worker."""
    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices available")

    sched = Scheduler(trace_set=test_trace_set, config=benchmark_config, devices=["cuda:0"])
    init_app(sched)
    yield sched
    sched.shutdown()


@pytest_asyncio.fixture
async def client(scheduler) -> AsyncClient:
    """Async HTTP client for testing the API."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
