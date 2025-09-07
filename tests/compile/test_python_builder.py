import torch

from flashinfer_bench.compile.builders import PythonBuilder
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages


def test_python_builder_minimum(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    d = Definition(
        name="mm",
        type="op",
        axes={"M": AxisConst(value=2), "N": AxisConst(value=2)},
        inputs={
            "A": TensorSpec(shape=["M", "N"], dtype="float32"),
            "B": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="import torch\n\ndef run(A, B):\n    return A",
    )
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="pkg/main.py::run"
    )
    srcs = [
        SourceFile(path="pkg/main.py", content="def run(A, B):\n    return A"),
    ]
    s = Solution(name="py_sol", definition="mm", author="me", spec=spec, sources=srcs)

    b = PythonBuilder()
    r = b.build(d, s)
    # Call runnable with torch tensors
    A = [[1, 2], [3, 4]]
    B = [[0, 0], [0, 0]]
    out = r(A=A, B=B)
    assert out == A
    # Ensure temp_dir recorded under our cache
    assert r.meta.get("temp_dir")
    assert str(cache_dir) in r.meta["temp_dir"]
    # Cleanup
    b.clear_cache()


def test_python_builder_add(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    defn = Definition(
        name="add",
        type="op",
        axes={"M": AxisConst(value=2), "N": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="import torch\n\ndef run(X, Y):\n    return X + Y",
    )
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
    )
    srcs = [
        SourceFile(
            path="main.py",
            content="import torch\n\ndef run(X: torch.Tensor, Y: torch.Tensor):\n    return X + Y",
        )
    ]
    sol = Solution(name="add_py", definition="add", author="tester", spec=spec, sources=srcs)

    # Build and run with torch tensors
    b = PythonBuilder()
    r = b.build(defn, sol)
    X = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    Y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    out = r(X=X, Y=Y)
    expected = torch.tensor([[6, 8], [10, 12]], dtype=torch.float32)
    assert torch.allclose(out, expected)
    b.clear_cache()
