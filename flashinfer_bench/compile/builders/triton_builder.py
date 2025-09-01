from __future__ import annotations

from flashinfer_bench.compile.builder import Builder, Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import Solution, SupportedLanguages


class TritonBuilder(Builder):
    def can_build(self, sol: Solution) -> bool:
        return sol.spec.language == SupportedLanguages.TRITON

    def fingerprint(self, sol: Solution) -> str:
        return f"triton::{sol.name}"

    def build(self, defn: Definition, sol: Solution) -> Runnable:
        raise NotImplementedError
