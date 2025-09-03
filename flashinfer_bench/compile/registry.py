from __future__ import annotations

from typing import List

from flashinfer_bench.compile.builder import Builder, BuildError
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages


class BuilderRegistry:
    """Registry that dispatches to the first capable builder."""

    def __init__(self) -> None:
        self._builders: List[Builder] = []
        self._frozen: bool = False

    def register(self, builder: Builder) -> None:
        if self._frozen:
            raise BuildError("Cannot register builder after registry is frozen")
        self._builders.append(builder)

    def freeze(self) -> None:
        if not self._frozen:
            self._builders = tuple(self._builders)
            self._frozen = True

    def clear(self) -> None:
        for b in self._builders:
            try:
                b.clear_cache()
            except Exception:
                pass
        self._builders.clear()

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

    @classmethod
    def default(cls) -> "BuilderRegistry":
        from flashinfer_bench.compile.builders import CUDABuilder, PythonBuilder, TritonBuilder

        reg = cls()
        reg.register(PythonBuilder())
        reg.register(TritonBuilder())
        reg.register(CUDABuilder())
        reg.freeze()
        return reg


_registry: BuilderRegistry | None = None


def get_registry() -> BuilderRegistry:
    global _registry
    if _registry is None:
        _registry = BuilderRegistry.default()
    return _registry
