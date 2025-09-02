from __future__ import annotations

from flashinfer_bench.compile.builder import Builder, BuildError, create_pkg_name
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import Solution, SupportedLanguages

from .python_builder import PythonBuilder


class TritonBuilder(Builder):
    def __init__(self) -> None:
        super().__init__()
        self._py_builder = PythonBuilder()

    def can_build(self, sol: Solution) -> bool:
        return sol.spec.language == SupportedLanguages.TRITON

    def _make_key(self, solution: Solution) -> str:
        return f"triton::{create_pkg_name(solution)}"

    def _make_closer(self, *args, **kwargs):
        raise NotImplementedError("Triton uses PythonBuilder's closer through _build")

    def _build(self, defn: Definition, sol: Solution) -> Runnable:
        # Triton availability check
        try:
            import triton as _triton
        except Exception as e:
            raise BuildError("Triton is not available in the current environment") from e
        # Reuse Python builder for source layout and import
        runnable = self._py_builder._build(defn, sol)
        runnable.meta.update(
            {"language": "triton", "triton_version": getattr(_triton, "__version__", None)}
        )
        return runnable
