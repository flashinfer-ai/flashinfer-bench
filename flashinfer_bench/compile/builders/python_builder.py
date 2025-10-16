from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from flashinfer_bench.compile.builder import (
    Builder,
    BuildError,
    create_pkg_name,
    write_sources_to_temp,
)
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data import Definition, Solution, SupportedLanguages


class PythonBuilder(Builder):
    """Load a Python entry point from provided sources into a temporary module."""

    def can_build(self, sol: Solution) -> bool:
        return sol.spec.language == SupportedLanguages.PYTHON

    def _make_key(self, solution: Solution) -> str:
        return f"python::{create_pkg_name(solution)}"

    def _make_closer(self, pkg: str, tmpdir: str) -> Callable[[], None]:
        def closer() -> None:
            try:
                # Unload module and submodules
                to_delete = [m for m in list(sys.modules) if m == pkg or m.startswith(pkg + ".")]
                for m in to_delete:
                    sys.modules.pop(m, None)
            except Exception:
                pass
            try:
                while tmpdir in sys.path:
                    try:
                        sys.path.remove(tmpdir)
                    except ValueError:
                        break
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        return closer

    def _build(self, defn: Definition, sol: Solution) -> Runnable:
        entry = sol.spec.entry_point
        try:
            entry_file, entry_func = entry.split("::", 1)
        except ValueError as e:
            raise BuildError("entry_point must be '<file.py>::<function>' for Python") from e

        # _fib_py_some_solution_<hash>
        pkg = create_pkg_name(sol, "fib_py_")
        # _fib_py_some_solution_<hash>.entry_file
        module_name = pkg + "." + ".".join(Path(entry_file).with_suffix("").parts)
        # $HOME/.cache/flashinfer_bench/python/<temp_dir>/<pkg>
        cache_root = os.environ.get(
            "FIB_CACHE_PATH", os.path.join(os.path.expanduser("~"), ".cache", "flashinfer_bench")
        )
        pkg_dir = write_sources_to_temp(
            base=os.path.join(cache_root, "python"), sources=sol.sources, pkg=pkg
        )
        tmp_root = os.path.dirname(pkg_dir)
        closer = self._make_closer(pkg, tmp_root)

        # Insert tmp_root into sys.path for import resolution
        sys.path.insert(0, tmp_root)

        if not os.path.exists(os.path.join(pkg_dir, *Path(entry_file).parts)):
            closer()
            raise BuildError(f"Entry file '{entry_file}' not found under tmp_root: {tmp_root}")

        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            closer()
            raise BuildError(f"Failed importing module '{module_name}' from sources: {e}") from e

        try:
            fn: Any = getattr(mod, entry_func)
        except AttributeError as e:
            closer()
            raise BuildError(
                f"Entry function '{entry_func}' not found in module '{module_name}'"
            ) from e

        if not callable(fn):
            closer()
            raise BuildError(f"Entry '{entry_func}' is not callable")

        meta = {
            "definition": defn.name,
            "solution": sol.name,
            "language": "python",
            "module": module_name,
            "entry": entry,
            "temp_dir": tmp_root,
        }

        return Runnable(fn=fn, closer=closer, meta=meta)
