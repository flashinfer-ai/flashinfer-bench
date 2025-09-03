from __future__ import annotations

from typing import Tuple

from flashinfer_bench.compile import builder
from flashinfer_bench.compile.builder import Builder, BuildError
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages


class BuilderRegistry:
    """Registry that dispatches to the first capable builder."""

    def __init__(self, builders: Tuple[Builder, ...]) -> None:
        if not builders:
            raise ValueError("BuilderRegistry requires at least one builder")
        self._builders: Tuple[Builder, ...] = builder

    def clear(self) -> None:
        for b in self._builders:
            try:
                b.clear_cache()
            except Exception:
                pass

    def build(self, defn: Definition, sol: Solution) -> Runnable:
        for builder in self._builders:
            # Choose the first
            if builder.can_build(sol):
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


def get_registry() -> BuilderRegistry:
    global _registry
    if _registry is None:
        from flashinfer_bench.compile.builders import CUDABuilder, PythonBuilder, TritonBuilder

        py = PythonBuilder()
        triton = TritonBuilder(py_builder=py)
        cuda = CUDABuilder()

        _registry = BuilderRegistry((py, triton, cuda))
    return _registry
