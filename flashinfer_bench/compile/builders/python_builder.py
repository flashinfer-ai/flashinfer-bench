from __future__ import annotations

from typing import Any

from flashinfer_bench.compile.builder import Builder, Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import Solution, SupportedLanguages


class PythonBuilder(Builder):
    """Placeholder: load entry from Python source."""

    def can_build(self, sol: Solution) -> bool:
        return sol.spec.language == SupportedLanguages.PYTHON

    def fingerprint(self, sol: Solution) -> str:
        return f"python::{sol.name}"

    def build(self, defn: Definition, sol: Solution) -> Runnable:
        raise NotImplementedError
