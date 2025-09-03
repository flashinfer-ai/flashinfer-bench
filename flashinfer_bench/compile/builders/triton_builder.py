from __future__ import annotations

from flashinfer_bench.compile.builder import Builder, BuildError, create_pkg_name
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import Solution, SupportedLanguages

from .python_builder import PythonBuilder


def _verify_triton() -> bool:
    try:
        import triton
    except Exception:
        return False
    return True


class TritonBuilder(Builder):
    _triton_available: bool = None

    @classmethod
    def _get_triton_available(cls) -> bool:
        if cls._triton_available is None:
            cls._triton_available = _verify_triton()
        return cls._triton_available

    def __init__(self, py_builder: PythonBuilder) -> None:
        super().__init__()
        self._py_builder = py_builder

    def can_build(self, sol: Solution) -> bool:
        return sol.spec.language == SupportedLanguages.TRITON and self._get_triton_available()

    def _make_key(self, solution: Solution) -> str:
        return f"triton::{create_pkg_name(solution)}"

    def _make_closer(self, *args, **kwargs):
        raise NotImplementedError("Triton uses PythonBuilder's closer through _build")

    def _build(self, defn: Definition, sol: Solution) -> Runnable:
        if not self._get_triton_available():
            raise BuildError("Triton is not available in the current environment")

        import triton

        # Reuse Python builder for source layout and import
        runnable = self._py_builder._build(defn, sol)
        runnable.meta.update({"language": "triton", "triton_version": triton.__version__})
        return runnable
