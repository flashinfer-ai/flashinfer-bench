from __future__ import annotations

from typing import Tuple

from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, SupportedLanguages

from .builder import Builder, BuildError
from .runnable import Runnable


class BuilderRegistry:
    """Registry that dispatches to the first capable builder."""

    def __init__(self, builders: Tuple[Builder, ...]) -> None:
        if not builders:
            raise ValueError("BuilderRegistry requires at least one builder")
        self._builders: Tuple[Builder, ...] = builders
        # Cache: solution_name -> builder to avoid repeated can_build checks
        self._solution_to_builder: dict[str, Builder] = {}

    def clear(self) -> None:
        for b in self._builders:
            try:
                b.clear_cache()
            except Exception:
                pass
        self._solution_to_builder.clear()

    def build(self, defn: Definition, sol: Solution) -> Runnable:
        builder = self._solution_to_builder.get(sol.name)
        if builder is not None:
            return builder.build(defn, sol)

        for builder in self._builders:
            if builder.can_build(sol):
                self._solution_to_builder[sol.name] = builder
                return builder.build(defn, sol)
        raise BuildError(f"No registered builder can build solution '{sol.name}'")

    def build_reference(self, defn: Definition) -> Runnable:
        pseudo = Solution(
            name=f"{defn.name}__reference",
            definition=defn.name,
            author="__builtin__",
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content=defn.reference)],
            description="reference",
        )
        return self.build(defn, pseudo)


_registry: BuilderRegistry | None = None


def get_builder_registry() -> BuilderRegistry:
    global _registry
    if _registry is None:
        from .builders import CUDABuilder, PythonBuilder, TritonBuilder, TVMFFIBuilder

        py = PythonBuilder()
        triton = TritonBuilder(py_builder=py)
        tvm_ffi = TVMFFIBuilder()
        cuda = CUDABuilder()  # Fallback for backward compatibility

        # Priority: Python > Triton > TVM-FFI > CUDA (pybind11)
        _registry = BuilderRegistry((py, triton, tvm_ffi, cuda))
    return _registry
