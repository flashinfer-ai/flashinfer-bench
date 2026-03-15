"""Generate GPU-Mode problem directories from flashinfer-trace Definitions.

Usage:
    python -m flashinfer_bench.external.gpumode.generate \
        --trace-path tmp-repos/flashinfer-trace \
        --output /tmp/test-problems
"""

import argparse
import ast
from pathlib import Path

import yaml
from jinja2 import Template

from flashinfer_bench.data import Definition, TraceSet

DESCRIPTION_TEMPLATE = Template(
    """\
Optimize {{ d.name }} kernel.
{% if d.description %}
{{ d.description }}
{% endif %}
Inputs:
{% for name, spec in d.inputs.items() %}  - `{{ name }}`: {{ spec.dtype.value }} {{ render_shape(d, spec.shape) }}
{% endfor %}Outputs:
{% for name, spec in d.outputs.items() %}  - `{{ name }}`: {{ spec.dtype.value }} {{ render_shape(d, spec.shape) }}
{% endfor %}
Part of the FlashInfer-Bench challenge series.
https://github.com/flashinfer-ai/flashinfer-bench/
"""
)


def render_shape(definition: Definition, shape: list[str] | None) -> str:
    """Render a symbolic shape to a human-readable string like [batch_size, 4096]."""
    if shape is None:
        return "scalar"
    parts = []
    for axis_name in shape:
        ax = definition.axes[axis_name]
        parts.append(str(ax.value) if ax.type == "const" else axis_name)
    return "[" + ", ".join(parts) + "]"


def render_description(definition: Definition) -> str:
    return DESCRIPTION_TEMPLATE.render(d=definition, render_shape=render_shape)


def make_task_yml(definition: Definition) -> str:
    task = {
        "files": [
            {"name": "submission.py", "source": "@SUBMISSION@"},
            {"name": "eval.py", "source": "../eval.py"},
            {"name": "reference.py", "source": "reference.py"},
        ],
        "lang": "py",
        "description": render_description(definition),
        "config": {"main": "eval.py"},
        "templates": {"Python": "submission.py"},
        "test_timeout": 300,
        "benchmark_timeout": 900,
        "ranked_timeout": 1200,
        "tests": [{"definition": definition.name, "mode": "test"}],
        "benchmarks": [{"definition": definition.name, "mode": "benchmark"}],
        "ranking_by": "geom",
    }
    return yaml.dump(task, default_flow_style=False, sort_keys=False, allow_unicode=True)


class _RunToCustomKernel(ast.NodeTransformer):
    """Rename 'def run(...)' to 'def custom_kernel(...)' and strip decorators."""

    def visit_FunctionDef(self, node):
        if node.name == "run":
            node.name = "custom_kernel"
            node.decorator_list = []
        return node


def _rename_run_to_custom_kernel(source: str) -> str:
    """Rename 'def run(...)' to 'def custom_kernel(...)' and strip decorators."""
    tree = ast.parse(source)
    tree = _RunToCustomKernel().visit(tree)
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    return '"""Optimize this kernel. Signature matches Definition inputs."""\n' + code + "\n"


def make_submission_py(definition: Definition) -> str:
    return _rename_run_to_custom_kernel(definition.reference)


def make_reference_py(definition: Definition) -> str:
    return definition.reference + "\n"


def make_competition_yaml(definitions: list[Definition], gpus: list[str]) -> str:
    competition = {
        "name": "flashinfer-bench",
        "description": "FlashInfer-Bench GPU kernel optimization challenges",
        "deadline": "2027-01-01",
        "problems": [
            {"name": f"fib-{d.name}", "directory": f"flashinfer-bench/{d.name}", "gpus": gpus}
            for d in definitions
        ],
    }
    return yaml.dump(competition, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate(trace_path: str, output: str, gpus: list[str] | None = None):
    """Generate problem directories matching reference-kernels layout.

    Output structure (mirrors problems/ in reference-kernels):
        output/
        ├── flashinfer-bench.yaml
        └── flashinfer-bench/
            ├── eval.py
            ├── rmsnorm_h4096/  (task.yml, submission.py, reference.py)
            └── ...
    """
    gpus = gpus or ["B200"]
    out = Path(output)
    problems_dir = out / "flashinfer-bench"
    problems_dir.mkdir(parents=True, exist_ok=True)

    trace_set = TraceSet.from_path(trace_path)
    definitions = sorted(trace_set.definitions.values(), key=lambda d: d.name)

    for d in definitions:
        d_dir = problems_dir / d.name
        d_dir.mkdir(parents=True, exist_ok=True)
        (d_dir / "task.yml").write_text(make_task_yml(d))
        (d_dir / "submission.py").write_text(make_submission_py(d))
        (d_dir / "reference.py").write_text(make_reference_py(d))

    (out / "flashinfer-bench.yaml").write_text(make_competition_yaml(definitions, gpus))

    eval_src = Path(__file__).parent / "eval.py"
    (problems_dir / "eval.py").write_text(eval_src.read_text())

    print(f"Generated {len(definitions)} problem directories in {problems_dir}")
    return definitions


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPU-Mode problem dirs from flashinfer-trace"
    )
    parser.add_argument("--trace-path", required=True, help="Path to flashinfer-trace dataset")
    parser.add_argument("--output", required=True, help="Output directory for problem dirs")
    parser.add_argument("--gpus", nargs="+", default=["B200"], help="GPU types for competition")
    args = parser.parse_args()
    generate(args.trace_path, args.output, args.gpus)


if __name__ == "__main__":
    main()
