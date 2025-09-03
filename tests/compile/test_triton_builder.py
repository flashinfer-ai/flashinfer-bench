import importlib

import pytest

from flashinfer_bench.compile.builder import BuildError
from flashinfer_bench.compile.builders import TritonBuilder
from flashinfer_bench.compile.builders.python_builder import PythonBuilder
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages


def minimal_def():
    return Definition(
        name="d",
        type="op",
        axes={"M": AxisConst(value=1)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A",
    )


def test_triton_builder_import_guard(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FLASHINFER_BENCH_CACHE_DIR", str(cache_dir))

    b = TritonBuilder(PythonBuilder())
    d = minimal_def()
    spec = BuildSpec(
        language=SupportedLanguages.TRITON, target_hardware=["gpu"], entry_point="main.py::run"
    )
    srcs = [SourceFile(path="main.py", content="import torch\n\ndef run(A):\n    return A")]
    s = Solution(name="tri_sol", definition="d", author="a", spec=spec, sources=srcs)

    # Mock the import to make triton unavailable
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("Mocked: triton not available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(BuildError, match="Triton is not available"):
        b.build(d, s)


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton not available")
def test_triton_builder_minimum(tmp_path, monkeypatch):
    # Reset cached availability in case previous tests mocked imports
    from flashinfer_bench.compile.builders import TritonBuilder as _TB

    monkeypatch.setattr(_TB, "_triton_available", None, raising=False)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FLASHINFER_BENCH_CACHE_DIR", str(cache_dir))

    b = TritonBuilder(PythonBuilder())
    d = minimal_def()
    spec = BuildSpec(
        language=SupportedLanguages.TRITON, target_hardware=["gpu"], entry_point="m/main.py::run"
    )
    srcs = [
        SourceFile(path="m/main.py", content="import torch\n\ndef run(A):\n    return A"),
    ]
    s = Solution(name="tri_ok", definition="d", author="a", spec=spec, sources=srcs)
    r = b.build(d, s)
    out = r(A=[1, 2, 3])
    assert out == [1, 2, 3]


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton not available")
def test_triton_vector_add(tmp_path, monkeypatch):
    # Reset cached availability in case previous tests mocked imports
    from flashinfer_bench.compile.builders import TritonBuilder as _TB

    monkeypatch.setattr(_TB, "_triton_available", None, raising=False)
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FLASHINFER_BENCH_CACHE_DIR", str(cache_dir))

    defn = Definition(
        name="vec_add",
        type="op",
        axes={"N": AxisConst(value=256)},
        inputs={
            "X": TensorSpec(shape=["N"], dtype="float32"),
            "Y": TensorSpec(shape=["N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(X, Y):\n    return X + Y",
    )

    triton_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)

def run(X, Y):
    n = X.numel()
    Z = torch.empty_like(X)
    BLOCK = 128
    grid = lambda meta: ( (n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    add_kernel[grid](X, Y, Z, n, BLOCK_SIZE=BLOCK)
    return Z
"""

    spec = BuildSpec(
        language=SupportedLanguages.TRITON, target_hardware=["gpu"], entry_point="m/main.py::run"
    )
    srcs = [
        SourceFile(path="m/main.py", content=triton_code),
    ]
    sol = Solution(
        name="triton_vec_add", definition="vec_add", author="tester", spec=spec, sources=srcs
    )

    b = TritonBuilder(PythonBuilder())
    r = b.build(defn, sol)
    X = torch.arange(256, dtype=torch.float32, device="cuda")
    Y = 2 * torch.ones(256, dtype=torch.float32, device="cuda")
    Z = r(X=X, Y=Y)
    assert torch.allclose(Z, X + Y)
