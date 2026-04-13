"""
Dataset correctness and completeness validator.

Checks performed (controlled by `checks` parameter):

[layout]
  - Duplicate definition names across op_types (error if two definitions share a stem)
  - definitions/<op_type>/<definition>.json: name and op_type fields match path
  - workloads/<op_type>/<definition>.jsonl: op_type matches path, definition field
    inside each trace entry matches path
  - solutions/<author>/<op_type>/<definition>/<solution>.json: author, op_type,
    definition, and name fields all match their path components
  - traces/<author>/<op_type>/<definition>.jsonl: op_type matches path, definition
    field inside each trace entry matches path
  - blob files referenced by safetensors workload inputs exist under blob/

[definition]
  - JSON conforms to Definition pydantic schema, which also validates:
    - reference is valid Python syntax with a top-level `run` function
    - all input/output shape axis references exist in the axes dict
    - constraints are valid Python expressions
    - input and output names do not overlap
  - definition description is non-empty (warn if missing)
  - all axes have non-empty description (warn if missing)
  - all inputs and outputs have non-empty description (warn if missing)
  - (GPU only) build reference implementation via BuilderRegistry

[workload]
  - Each JSONL line conforms to Trace pydantic schema
  - Each entry must be workload-only (no solution or evaluation)
  - All variable axes from definition are provided, no extra axes, all values > 0
  - Input names exactly match definition (no missing, no extra)
  - Input types are consistent with definition (scalar vs tensor)
  - Tensor shapes match definition's shape inference (const axes + workload axes)
  - Dumped tensors (safetensors): blob exists, tensor key present, shape matches

[solution]
  - JSON conforms to Solution pydantic schema, which also validates:
    - entry_point follows the `{file_path}::{function_name}` format
    - entry source file exists in the sources list
    - sources list is non-empty
  - definition, author, and name fields all match their path components

[trace]
  - Each JSONL line conforms to Trace pydantic schema
  - Each entry must have solution and evaluation (not workload-only)
  - definition field inside each trace entry matches the expected definition name
  - Referenced solutions must exist in the solution index (error if unknown)
  - Per-author workload coverage vs the full workload set (warn if incomplete)

[baseline]
  - Baseline solution exists under solutions/baseline/ (warn if missing)
  - (GPU only) build each baseline solution via BuilderRegistry
  - Baseline trace exists under traces/baseline/ (warn if missing)
  - Baseline trace solution fields map to solutions/baseline/ (error if not)
  - All baseline trace entries have PASSED evaluation status (warn if not)
  - Baseline trace covers all workloads for the definition (warn if incomplete)

[benchmark] (requires GPU, skipped with disable_gpu=True)
  - Package definition reference code as pseudo-solution
  - Run baseline solutions + reference via Benchmark.run_all()
  - All workloads must pass for both baseline and reference
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from safetensors import safe_open

from flashinfer_bench.bench.benchmark import Benchmark
from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.compile.builder import BuildError
from flashinfer_bench.compile.registry import BuilderRegistry
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages
from flashinfer_bench.data.trace import EvaluationStatus, Trace
from flashinfer_bench.data.trace_set import TraceSet
from flashinfer_bench.data.validate_render import (
    CheckMessage,
    CheckResult,
    DatasetReport,
    DefinitionReport,
    ReportConfig,
    compute_definition_status,
    render_text_report,
)
from flashinfer_bench.data.workload import RandomInput, SafetensorsInput, ScalarInput
from flashinfer_bench.env import get_fib_dataset_path

logger = logging.getLogger(__name__)

ALL_CHECKS = ["layout", "definition", "workload", "solution", "trace", "baseline", "benchmark"]
CPU_CHECKS = ["layout", "definition", "workload", "solution", "trace", "baseline"]


# ---------------------------------------------------------------------------
# Scanning data structures
# ---------------------------------------------------------------------------


@dataclass
class ScannedDefinition:
    op_type: str
    path: Path
    definition: Optional[Definition] = None
    error: Optional[str] = None


@dataclass
class ScannedWorkload:
    op_type: str
    path: Path
    traces: Optional[list] = None
    error: Optional[str] = None


@dataclass
class ScannedSolution:
    author: str
    op_type: str
    definition_name: str
    solution_name: str
    path: Path
    solution: Optional[Solution] = None
    error: Optional[str] = None


@dataclass
class ScannedTrace:
    author: str
    op_type: str
    definition_name: str
    path: Path
    traces: Optional[list] = None
    error: Optional[str] = None


@dataclass
class DatasetIndex:
    definitions: dict[str, ScannedDefinition] = field(default_factory=dict)
    workloads: dict[str, ScannedWorkload] = field(default_factory=dict)
    solutions: dict[str, list[ScannedSolution]] = field(default_factory=dict)
    trace_files: dict[tuple[str, str], ScannedTrace] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(messages: list[CheckMessage]) -> CheckResult:
    """Build a CheckResult with status derived from the worst message level.

    Parameters
    ----------
    messages : list[CheckMessage]
        Collected check messages.

    Returns
    -------
    CheckResult
        Result with status="error" if any error, "warning" if any warning, else "ok".
    """
    if any(m.level == "error" for m in messages):
        status = "error"
    elif any(m.level == "warning" for m in messages):
        status = "warning"
    else:
        status = "ok"
    return CheckResult(status=status, messages=messages)


def _load_json(model_class, path: Path):
    """Load a pydantic model from a JSON file. Returns (instance, None) or (None, error_str)."""
    try:
        return model_class.model_validate_json(path.read_text()), None
    except Exception as exc:
        return None, str(exc)


def _load_jsonl(model_class, path: Path):
    """Load a list of pydantic models from JSONL. Returns (list, None) or (None, error_str)."""
    items = []
    line_number = 0
    try:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            items.append(model_class.model_validate_json(line))
        return items, None
    except Exception as exc:
        suffix = f" (line {line_number})" if line_number else ""
        return None, f"{exc}{suffix}"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def scan_dataset(dataset_root: Path) -> DatasetIndex:
    """Scan all dataset directories and build an index with loaded content.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the FlashInfer Trace dataset.

    Returns
    -------
    DatasetIndex
        Index containing all scanned definitions, workloads, solutions, and traces.
    """
    index = DatasetIndex()

    definitions_dir = dataset_root / "definitions"
    if definitions_dir.is_dir():
        for path in sorted(definitions_dir.rglob("*.json")):
            relative = path.relative_to(definitions_dir)
            if len(relative.parts) != 2:
                continue
            op_type = relative.parts[0]
            definition_name = relative.stem
            definition, error = _load_json(Definition, path)
            if definition_name in index.definitions:
                existing = index.definitions[definition_name]
                error = f"duplicate definition name: {existing.path} and {path}"
            index.definitions[definition_name] = ScannedDefinition(
                op_type=op_type, path=path, definition=definition, error=error
            )

    workloads_dir = dataset_root / "workloads"
    if workloads_dir.is_dir():
        for path in sorted(workloads_dir.rglob("*.jsonl")):
            relative = path.relative_to(workloads_dir)
            if len(relative.parts) != 2:
                continue
            op_type = relative.parts[0]
            definition_name = relative.stem
            traces, error = _load_jsonl(Trace, path)
            index.workloads[definition_name] = ScannedWorkload(
                op_type=op_type, path=path, traces=traces, error=error
            )

    solutions_dir = dataset_root / "solutions"
    if solutions_dir.is_dir():
        for path in sorted(solutions_dir.rglob("*.json")):
            relative = path.relative_to(solutions_dir)
            if len(relative.parts) != 4:
                continue
            author, op_type, definition_name, filename = relative.parts
            solution_name = Path(filename).stem
            solution, error = _load_json(Solution, path)
            index.solutions.setdefault(definition_name, []).append(
                ScannedSolution(
                    author=author,
                    op_type=op_type,
                    definition_name=definition_name,
                    solution_name=solution_name,
                    path=path,
                    solution=solution,
                    error=error,
                )
            )

    traces_dir = dataset_root / "traces"
    if traces_dir.is_dir():
        for path in sorted(traces_dir.rglob("*.jsonl")):
            relative = path.relative_to(traces_dir)
            if len(relative.parts) != 3:
                continue
            author, op_type, filename = relative.parts
            definition_name = Path(filename).stem
            traces, error = _load_jsonl(Trace, path)
            index.trace_files[(author, definition_name)] = ScannedTrace(
                author=author,
                op_type=op_type,
                definition_name=definition_name,
                path=path,
                traces=traces,
                error=error,
            )

    return index


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_definitions(
    index: DatasetIndex, requested_op_types: list[str], requested_definitions: list[str]
) -> list[str]:
    """Return definition names matching the filter (union logic).

    If both lists are empty, returns all definitions. Otherwise returns
    definitions matching any of the requested op_types OR definition names.

    Parameters
    ----------
    index : DatasetIndex
        Scanned dataset index.
    requested_op_types : list[str]
        Op types to include.
    requested_definitions : list[str]
        Definition names to include.

    Returns
    -------
    list[str]
        Sorted list of matching definition names.
    """
    if not requested_op_types and not requested_definitions:
        return sorted(index.definitions.keys())

    result = set()
    for definition_name, entry in index.definitions.items():
        if requested_op_types and entry.op_type in requested_op_types:
            result.add(definition_name)
        if requested_definitions and definition_name in requested_definitions:
            result.add(definition_name)
    return sorted(result)


# ---------------------------------------------------------------------------
# Check: layout
# ---------------------------------------------------------------------------


def check_layout(
    dataset_root: Path,
    definition_name: str,
    definition_entry: ScannedDefinition,
    workload_entry: Optional[ScannedWorkload],
    solution_entries: list[ScannedSolution],
    trace_entries: list[ScannedTrace],
) -> CheckResult:
    """Verify directory structure and path-field consistency for one definition.

    Parameters
    ----------
    dataset_root : Path
        Dataset root directory.
    definition_name : str
        Name of the definition being checked.
    definition_entry : ScannedDefinition
        Scanned definition data.
    workload_entry : Optional[ScannedWorkload]
        Scanned workload data, or None if no workload file exists.
    solution_entries : list[ScannedSolution]
        All solutions associated with this definition.
    trace_entries : list[ScannedTrace]
        All trace files associated with this definition.

    Returns
    -------
    CheckResult
        Result with any path/field mismatches or missing blob errors.
    """
    messages: list[CheckMessage] = []
    definition = definition_entry.definition

    if definition_entry.error and definition_entry.error.startswith("duplicate definition name"):
        messages.append(CheckMessage(level="error", message=definition_entry.error))
        return make_result(messages)

    if definition is not None:
        if definition.name != definition_name:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"definition name mismatch: path says '{definition_name}', JSON says '{definition.name}'",
                )
            )
        if definition.op_type != definition_entry.op_type:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"definition op_type mismatch: path says '{definition_entry.op_type}', JSON says '{definition.op_type}'",
                )
            )

    if workload_entry is not None:
        if workload_entry.op_type != definition_entry.op_type:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"workload op_type mismatch: expected '{definition_entry.op_type}', got '{workload_entry.op_type}'",
                )
            )
        if workload_entry.traces is not None:
            for trace in workload_entry.traces:
                if trace.definition != definition_name:
                    messages.append(
                        CheckMessage(
                            level="error",
                            message=f"workload trace definition mismatch: expected '{definition_name}', got '{trace.definition}'",
                        )
                    )
                    break

    for solution_entry in solution_entries:
        if solution_entry.solution is None:
            continue
        solution = solution_entry.solution
        if solution.author != solution_entry.author:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"solution '{solution_entry.solution_name}' author mismatch: path '{solution_entry.author}', JSON '{solution.author}'",
                )
            )
        if solution.definition != definition_name:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"solution '{solution_entry.solution_name}' definition mismatch: path '{definition_name}', JSON '{solution.definition}'",
                )
            )
        if solution.name != solution_entry.solution_name:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"solution name mismatch: path '{solution_entry.solution_name}', JSON '{solution.name}'",
                )
            )
        if solution_entry.op_type != definition_entry.op_type:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"solution '{solution_entry.solution_name}' op_type mismatch: expected '{definition_entry.op_type}', got '{solution_entry.op_type}'",
                )
            )

    for trace_entry in trace_entries:
        if trace_entry.op_type != definition_entry.op_type:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"trace file op_type mismatch for author '{trace_entry.author}': expected '{definition_entry.op_type}', got '{trace_entry.op_type}'",
                )
            )
        if trace_entry.traces is not None:
            for trace in trace_entry.traces:
                if trace.definition != definition_name:
                    messages.append(
                        CheckMessage(
                            level="error",
                            message=f"trace definition mismatch for author '{trace_entry.author}': expected '{definition_name}', got '{trace.definition}'",
                        )
                    )
                    break

    if workload_entry is not None and workload_entry.traces is not None and definition is not None:
        for workload_trace in workload_entry.traces:
            workload = workload_trace.workload
            for input_name, input_spec in workload.inputs.items():
                if isinstance(input_spec, SafetensorsInput):
                    blob_path = dataset_root / input_spec.path
                    if not blob_path.exists():
                        messages.append(
                            CheckMessage(
                                level="error",
                                message=f"workload {workload.uuid}: blob missing at {input_spec.path}",
                            )
                        )

    return make_result(messages)


# ---------------------------------------------------------------------------
# Check: definition
# ---------------------------------------------------------------------------


def check_definition_content(
    definition_name: str, definition_entry: ScannedDefinition, disable_gpu: bool = False
) -> CheckResult:
    """Validate definition schema, reference code, descriptions, and build.

    Parameters
    ----------
    definition_name : str
        Name of the definition being checked.
    definition_entry : ScannedDefinition
        Scanned definition data (with parsed Definition or parse error).
    disable_gpu : bool
        When False, also build the reference implementation to verify it compiles.

    Returns
    -------
    CheckResult
        Result with parse errors, missing-description warnings, or build errors.
    """
    messages: list[CheckMessage] = []

    if definition_entry.error is not None:
        messages.append(
            CheckMessage(level="error", message=f"parse error: {definition_entry.error}")
        )
        return make_result(messages)

    definition = definition_entry.definition

    if not definition.description:
        messages.append(CheckMessage(level="warning", message="missing description"))

    for axis_name, axis_spec in definition.axes.items():
        if not getattr(axis_spec, "description", None):
            messages.append(
                CheckMessage(level="warning", message=f"missing description: axes.{axis_name}")
            )

    for input_name, tensor_spec in definition.inputs.items():
        if not tensor_spec.description:
            messages.append(
                CheckMessage(level="warning", message=f"missing description: inputs.{input_name}")
            )

    for output_name, tensor_spec in definition.outputs.items():
        if not tensor_spec.description:
            messages.append(
                CheckMessage(level="warning", message=f"missing description: outputs.{output_name}")
            )

    if not disable_gpu:
        try:
            registry = BuilderRegistry.get_instance()
            registry.build_reference(definition)
            messages.append(CheckMessage(level="info", message="reference build: ok"))
        except (BuildError, Exception) as exc:
            messages.append(CheckMessage(level="error", message=f"reference build failed: {exc}"))

    return make_result(messages)


# ---------------------------------------------------------------------------
# Check: workload
# ---------------------------------------------------------------------------


def check_workload_content(
    definition_name: str,
    definition: Optional[Definition],
    workload_entry: Optional[ScannedWorkload],
    dataset_root: Path,
) -> CheckResult:
    """Validate workload schema, axes, shapes, and dumped tensors.

    Parameters
    ----------
    definition_name : str
        Name of the definition these workloads belong to.
    definition : Optional[Definition]
        Parsed definition, or None if parsing failed.
    workload_entry : Optional[ScannedWorkload]
        Scanned workload data, or None if no workload file exists.
    dataset_root : Path
        Dataset root directory (for resolving blob paths).

    Returns
    -------
    CheckResult
        Result with axes/shape/blob errors per workload entry.
    """
    messages: list[CheckMessage] = []

    if workload_entry is None:
        messages.append(CheckMessage(level="warning", message="no workload file found"))
        return make_result(messages)

    if workload_entry.error is not None:
        messages.append(CheckMessage(level="error", message=f"parse error: {workload_entry.error}"))
        return make_result(messages)

    if definition is None:
        messages.append(
            CheckMessage(
                level="error", message="cannot validate workloads: definition failed to load"
            )
        )
        return make_result(messages)

    valid_count = 0
    var_axis_names = set(definition.var_axes)
    definition_input_names = set(definition.inputs.keys())
    input_names_ordered = list(definition.inputs.keys())

    for workload_trace in workload_entry.traces:
        workload = workload_trace.workload
        uuid = workload.uuid
        has_error = False

        if not workload_trace.is_workload_trace():
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"{uuid}: workload entry should not have solution or evaluation",
                )
            )
            has_error = True

        provided_axes = set(workload.axes.keys())
        missing_axes = var_axis_names - provided_axes
        extra_axes = provided_axes - var_axis_names
        if missing_axes:
            messages.append(
                CheckMessage(level="error", message=f"{uuid}: missing axes {sorted(missing_axes)}")
            )
            has_error = True
        if extra_axes:
            messages.append(
                CheckMessage(level="error", message=f"{uuid}: extra axes {sorted(extra_axes)}")
            )
            has_error = True

        for axis_name, axis_value in workload.axes.items():
            if axis_value <= 0:
                messages.append(
                    CheckMessage(
                        level="error",
                        message=f"{uuid}: axis {axis_name} has non-positive value {axis_value}",
                    )
                )
                has_error = True

        workload_input_names = set(workload.inputs.keys())
        missing_inputs = definition_input_names - workload_input_names
        extra_inputs = workload_input_names - definition_input_names
        if missing_inputs:
            messages.append(
                CheckMessage(
                    level="error", message=f"{uuid}: missing inputs {sorted(missing_inputs)}"
                )
            )
            has_error = True
        if extra_inputs:
            messages.append(
                CheckMessage(level="error", message=f"{uuid}: extra inputs {sorted(extra_inputs)}")
            )
            has_error = True

        if has_error:
            continue

        try:
            expected_shapes = definition.get_input_shapes(workload.axes)
        except ValueError as exc:
            messages.append(
                CheckMessage(level="error", message=f"{uuid}: shape inference failed: {exc}")
            )
            continue

        for input_name, expected_shape in zip(input_names_ordered, expected_shapes):
            input_spec = workload.inputs[input_name]
            definition_tensor = definition.inputs[input_name]

            if isinstance(input_spec, ScalarInput):
                if definition_tensor.shape is not None:
                    messages.append(
                        CheckMessage(
                            level="error",
                            message=f"{uuid}: {input_name} is scalar but definition expects tensor",
                        )
                    )
                    has_error = True
            elif isinstance(input_spec, RandomInput):
                if definition_tensor.shape is None:
                    messages.append(
                        CheckMessage(
                            level="error",
                            message=f"{uuid}: {input_name} is random but definition expects scalar",
                        )
                    )
                    has_error = True
            elif isinstance(input_spec, SafetensorsInput):
                if definition_tensor.shape is None:
                    messages.append(
                        CheckMessage(
                            level="error",
                            message=f"{uuid}: {input_name} is safetensors but definition expects scalar",
                        )
                    )
                    has_error = True
                    continue
                blob_path = dataset_root / input_spec.path
                if not blob_path.exists():
                    messages.append(
                        CheckMessage(
                            level="error", message=f"{uuid}: blob missing at {input_spec.path}"
                        )
                    )
                    has_error = True
                    continue
                try:
                    with safe_open(str(blob_path), framework="pt") as f:
                        if input_spec.tensor_key not in f.keys():
                            messages.append(
                                CheckMessage(
                                    level="error",
                                    message=f"{uuid}: tensor key '{input_spec.tensor_key}' not found in {input_spec.path}",
                                )
                            )
                            has_error = True
                        else:
                            tensor = f.get_tensor(input_spec.tensor_key)
                            actual_shape = tuple(tensor.shape)
                            if expected_shape is not None and actual_shape != tuple(expected_shape):
                                messages.append(
                                    CheckMessage(
                                        level="error",
                                        message=f"{uuid}: {input_name} shape mismatch: expected {tuple(expected_shape)}, got {actual_shape}",
                                    )
                                )
                                has_error = True
                except Exception as exc:
                    messages.append(
                        CheckMessage(level="error", message=f"{uuid}: blob load failed: {exc}")
                    )
                    has_error = True

        if not has_error:
            valid_count += 1

    total = len(workload_entry.traces)
    if valid_count == total:
        messages.insert(0, CheckMessage(level="info", message=f"{valid_count} valid"))
    else:
        messages.insert(0, CheckMessage(level="info", message=f"{valid_count}/{total} valid"))

    return make_result(messages)


# ---------------------------------------------------------------------------
# Check: solution
# ---------------------------------------------------------------------------


def check_solution_content(
    definition_name: str, solution_entries: list[ScannedSolution]
) -> CheckResult:
    """Validate solution schemas and field consistency.

    Parameters
    ----------
    definition_name : str
        Name of the definition these solutions belong to.
    solution_entries : list[ScannedSolution]
        All solutions associated with this definition.

    Returns
    -------
    CheckResult
        Result with parse errors or field mismatches.
    """
    messages: list[CheckMessage] = []

    if not solution_entries:
        messages.append(CheckMessage(level="info", message="no solutions"))
        return make_result(messages)

    for entry in solution_entries:
        if entry.error is not None:
            messages.append(
                CheckMessage(
                    level="error", message=f"{entry.solution_name}: parse error: {entry.error}"
                )
            )
            continue
        solution = entry.solution
        if solution.definition != definition_name:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"{entry.solution_name}: definition field '{solution.definition}' != '{definition_name}'",
                )
            )
        if solution.author != entry.author:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"{entry.solution_name}: author field '{solution.author}' != path '{entry.author}'",
                )
            )
        if solution.name != entry.solution_name:
            messages.append(
                CheckMessage(
                    level="error",
                    message=f"{entry.solution_name}: name field '{solution.name}' != path '{entry.solution_name}'",
                )
            )

    valid_count = sum(1 for e in solution_entries if e.error is None)
    messages.insert(0, CheckMessage(level="info", message=f"{valid_count} solutions"))

    return make_result(messages)


# ---------------------------------------------------------------------------
# Check: trace
# ---------------------------------------------------------------------------


def check_trace_content(
    definition_name: str,
    trace_entries: list[ScannedTrace],
    workload_uuids: set[str],
    solution_names: set[str],
) -> CheckResult:
    """Validate trace structure and workload coverage per author.

    Parameters
    ----------
    definition_name : str
        Name of the definition these traces belong to.
    trace_entries : list[ScannedTrace]
        All trace files associated with this definition.
    workload_uuids : set[str]
        UUIDs of all known workloads for coverage comparison.
    solution_names : set[str]
        Names of all known solutions for reference validation.

    Returns
    -------
    CheckResult
        Result with parse errors, unknown solution warnings, or coverage gaps.
    """
    messages: list[CheckMessage] = []

    if not trace_entries:
        messages.append(CheckMessage(level="info", message="no trace files"))
        return make_result(messages)

    for trace_entry in trace_entries:
        author = trace_entry.author
        if trace_entry.error is not None:
            messages.append(
                CheckMessage(level="error", message=f"{author}: parse error: {trace_entry.error}")
            )
            continue

        for trace in trace_entry.traces:
            if trace.definition != definition_name:
                messages.append(
                    CheckMessage(
                        level="error",
                        message=f"{author}: trace definition '{trace.definition}' != '{definition_name}'",
                    )
                )
                break
            if trace.solution is None or trace.evaluation is None:
                messages.append(
                    CheckMessage(
                        level="error",
                        message=f"{author}: trace entry {trace.workload.uuid} missing solution or evaluation",
                    )
                )
                continue
            if trace.solution not in solution_names:
                messages.append(
                    CheckMessage(
                        level="error",
                        message=f"{author}: trace references unknown solution '{trace.solution}'",
                    )
                )

        if trace_entry.traces and workload_uuids:
            covered_uuids = {t.workload.uuid for t in trace_entry.traces}
            missing = workload_uuids - covered_uuids
            if missing:
                messages.append(
                    CheckMessage(
                        level="warning",
                        message=f"{author} missing {len(missing)}/{len(workload_uuids)} workloads",
                    )
                )

    return make_result(messages)


# ---------------------------------------------------------------------------
# Check: baseline
# ---------------------------------------------------------------------------


def check_baseline_content(
    definition_name: str,
    solution_entries: list[ScannedSolution],
    trace_entries: list[ScannedTrace],
    workload_uuids: set[str],
    definition: Optional[Definition] = None,
    disable_gpu: bool = False,
) -> CheckResult:
    """Validate baseline solution existence, build, and trace coverage/status.

    Parameters
    ----------
    definition_name : str
        Name of the definition being checked.
    solution_entries : list[ScannedSolution]
        All solutions (filtered to author="baseline" internally).
    trace_entries : list[ScannedTrace]
        All trace files (filtered to author="baseline" internally).
    workload_uuids : set[str]
        UUIDs of all known workloads for coverage comparison.
    definition : Optional[Definition]
        Parsed definition (needed for build checks).
    disable_gpu : bool
        When False, also build each baseline solution to verify it compiles.

    Returns
    -------
    CheckResult
        Result with warnings for missing baseline solution/trace, non-passing
        entries, incomplete workload coverage, or build errors.
    """
    messages: list[CheckMessage] = []

    baseline_solutions = [e for e in solution_entries if e.author == "baseline"]
    baseline_trace = next((t for t in trace_entries if t.author == "baseline"), None)

    if not baseline_solutions:
        messages.append(CheckMessage(level="warning", message="no baseline solution"))
    else:
        names = [e.solution_name for e in baseline_solutions]
        messages.append(CheckMessage(level="info", message=f"solution: {', '.join(names)}"))

        if not disable_gpu and definition is not None:
            registry = BuilderRegistry.get_instance()
            for entry in baseline_solutions:
                if entry.error is not None or entry.solution is None:
                    continue
                try:
                    registry.build(definition, entry.solution)
                    messages.append(
                        CheckMessage(level="info", message=f"build {entry.solution_name}: ok")
                    )
                except (BuildError, Exception) as exc:
                    messages.append(
                        CheckMessage(
                            level="error", message=f"build {entry.solution_name} failed: {exc}"
                        )
                    )

    if baseline_trace is None:
        if baseline_solutions:
            messages.append(CheckMessage(level="warning", message="trace missing"))
    elif baseline_trace.error is not None:
        messages.append(
            CheckMessage(level="error", message=f"trace parse error: {baseline_trace.error}")
        )
    else:
        baseline_solution_names = {e.solution_name for e in baseline_solutions}
        for trace in baseline_trace.traces:
            if trace.solution and trace.solution not in baseline_solution_names:
                messages.append(
                    CheckMessage(
                        level="error",
                        message=f"trace references non-baseline solution '{trace.solution}'",
                    )
                )
                break

        failed = [
            t
            for t in baseline_trace.traces
            if t.evaluation and t.evaluation.status != EvaluationStatus.PASSED
        ]
        if failed:
            messages.append(
                CheckMessage(
                    level="warning", message=f"trace has {len(failed)} non-passing entries"
                )
            )
        else:
            messages.append(CheckMessage(level="info", message="trace: all passed"))

        if workload_uuids:
            covered = {t.workload.uuid for t in baseline_trace.traces}
            missing = workload_uuids - covered
            if missing:
                messages.append(
                    CheckMessage(
                        level="warning",
                        message=f"trace covers {len(covered)}/{len(workload_uuids)} workloads",
                    )
                )
            else:
                messages.append(
                    CheckMessage(
                        level="info",
                        message=f"trace: {len(covered)}/{len(workload_uuids)} workloads",
                    )
                )

    return make_result(messages)


# ---------------------------------------------------------------------------
# Check: benchmark
# ---------------------------------------------------------------------------


def check_benchmark_content(
    definition_name: str,
    definition: Definition,
    workload_traces: list[Trace],
    baseline_solutions: list[Solution],
    dataset_root: Path,
) -> CheckResult:
    """Run baseline + reference through Benchmark.run_all and check results.

    Parameters
    ----------
    definition_name : str
        Name of the definition being benchmarked.
    definition : Definition
        Parsed definition (used to build the reference pseudo-solution).
    workload_traces : list[Trace]
        Workload traces to benchmark against.
    baseline_solutions : list[Solution]
        Baseline solutions to benchmark.
    dataset_root : Path
        Dataset root directory (for TraceSet construction).

    Returns
    -------
    CheckResult
        Result with benchmark pass/fail for both baseline and reference.
    """
    messages: list[CheckMessage] = []

    if not workload_traces:
        messages.append(CheckMessage(level="warning", message="no workloads to benchmark"))
        return make_result(messages)

    if not baseline_solutions:
        messages.append(
            CheckMessage(level="warning", message="no baseline solutions for benchmark")
        )
        return make_result(messages)

    reference_solution = Solution(
        name=f"{definition_name}__reference",
        definition=definition_name,
        author="__reference__",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cuda"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=definition.reference)],
    )

    all_solutions = list(baseline_solutions) + [reference_solution]

    trace_set = TraceSet(
        root=dataset_root,
        definitions={definition_name: definition},
        solutions={definition_name: all_solutions},
        workloads={definition_name: workload_traces},
        traces={},
    )

    config = BenchmarkConfig(
        warmup_runs=2, iterations=5, num_trials=1, definitions=[definition_name]
    )

    try:
        benchmark = Benchmark(trace_set, config)
        try:
            result_trace_set = benchmark.run_all(dump_traces=False)
        finally:
            benchmark.close()
    except Exception as exc:
        messages.append(CheckMessage(level="error", message=f"benchmark failed: {exc}"))
        return make_result(messages)

    result_traces = result_trace_set.traces.get(definition_name, [])
    baseline_names = {s.name for s in baseline_solutions}

    baseline_failures = []
    reference_failures = []
    for trace in result_traces:
        if not trace.is_successful():
            status = trace.evaluation.status.value if trace.evaluation else "no evaluation"
            if trace.solution in baseline_names:
                baseline_failures.append(f"{trace.solution} on {trace.workload.uuid}: {status}")
            elif trace.solution == reference_solution.name:
                reference_failures.append(f"workload {trace.workload.uuid}: {status}")

    if not baseline_failures:
        messages.append(CheckMessage(level="info", message="baseline: PASS"))
    else:
        for failure in baseline_failures:
            messages.append(CheckMessage(level="error", message=f"baseline failed: {failure}"))

    if not reference_failures:
        messages.append(CheckMessage(level="info", message="reference: PASS"))
    else:
        for failure in reference_failures:
            messages.append(CheckMessage(level="error", message=f"reference failed: {failure}"))

    return make_result(messages)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_dataset(
    dataset: Optional[str] = None,
    op_types: Optional[list[str]] = None,
    definitions: Optional[list[str]] = None,
    checks: Optional[
        list[
            Literal[
                "layout", "definition", "workload", "solution", "trace", "baseline", "benchmark"
            ]
        ]
    ] = None,
    disable_gpu: bool = False,
    output_folder: Optional[str] = None,
    outputs: Optional[list[Literal["stdout", "json", "text"]]] = None,
) -> DatasetReport:
    """Validate a FlashInfer Trace dataset and produce a report.

    Parameters
    ----------
    dataset : Optional[str]
        Dataset root directory. Defaults to FIB_DATASET_PATH env var.
    op_types : Optional[list[str]]
        Only check definitions under these op_types (union with ``definitions``).
    definitions : Optional[list[str]]
        Only check these definition names (union with ``op_types``).
    checks : Optional[list[Literal["layout", "definition", "workload", "solution", "trace", "baseline", "benchmark"]]]
        Check categories to run. Defaults to all categories (CPU-only when
        ``disable_gpu=True``, i.e. excludes "benchmark").
    disable_gpu : bool
        Skip benchmark checks (removes "benchmark" from checks).
    output_folder : Optional[str]
        Report output folder. Defaults to ``<dataset>/reports/``.
    outputs : Optional[list[Literal["stdout", "json", "text"]]]
        Output targets. Defaults to ``["stdout", "json", "text"]``.

    Returns
    -------
    DatasetReport
        The validation report with per-definition check results.
    """
    op_types = op_types or []
    definitions_filter = definitions or []
    outputs = outputs or ["stdout", "json", "text"]

    dataset_root = Path(dataset) if dataset else get_fib_dataset_path()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    if checks is None:
        checks = CPU_CHECKS if disable_gpu else ALL_CHECKS
    for check in checks:
        if check not in ALL_CHECKS:
            raise ValueError(f"unknown check '{check}', valid: {ALL_CHECKS}")
    if disable_gpu and "benchmark" in checks:
        checks = [c for c in checks if c != "benchmark"]

    for target in outputs:
        if target not in ("stdout", "json", "text"):
            raise ValueError(f"unknown output target '{target}', valid: stdout, json, text")
    if not outputs:
        raise ValueError("outputs cannot be empty")

    report_output_folder = Path(output_folder) if output_folder else dataset_root / "reports"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = report_output_folder / f"report-{timestamp}.json"
    text_path = report_output_folder / f"report-{timestamp}.txt"

    if "json" in outputs and json_path.exists():
        raise FileExistsError(f"report file already exists: {json_path}")
    if "text" in outputs and text_path.exists():
        raise FileExistsError(f"report file already exists: {text_path}")

    logger.info(f"Scanning dataset at {dataset_root}")
    index = scan_dataset(dataset_root)
    logger.info(f"Found {len(index.definitions)} definitions")

    target_definitions = filter_definitions(index, op_types, definitions_filter)
    logger.info(f"Target definitions: {len(target_definitions)}")

    report_config = ReportConfig(
        dataset=str(dataset_root),
        checks=checks,
        disable_gpu=disable_gpu,
        op_types=op_types,
        definitions=definitions_filter,
    )

    definition_reports: dict[str, DefinitionReport] = {}

    for definition_name in target_definitions:
        definition_entry = index.definitions[definition_name]
        workload_entry = index.workloads.get(definition_name)
        solution_entries = index.solutions.get(definition_name, [])
        trace_entries = [
            entry for key, entry in index.trace_files.items() if key[1] == definition_name
        ]

        definition = definition_entry.definition
        workload_traces = workload_entry.traces if workload_entry and workload_entry.traces else []
        workload_uuids = {t.workload.uuid for t in workload_traces}
        all_solution_names = {e.solution_name for e in solution_entries if e.solution is not None}

        report = DefinitionReport()

        if "layout" in checks:
            report.layout = check_layout(
                dataset_root,
                definition_name,
                definition_entry,
                workload_entry,
                solution_entries,
                trace_entries,
            )

        if "definition" in checks:
            report.definition = check_definition_content(
                definition_name, definition_entry, disable_gpu=disable_gpu
            )

        if "workload" in checks:
            report.workload = check_workload_content(
                definition_name, definition, workload_entry, dataset_root
            )

        if "solution" in checks:
            report.solution = check_solution_content(definition_name, solution_entries)

        if "trace" in checks:
            report.trace = check_trace_content(
                definition_name, trace_entries, workload_uuids, all_solution_names
            )

        if "baseline" in checks:
            report.baseline = check_baseline_content(
                definition_name,
                solution_entries,
                trace_entries,
                workload_uuids,
                definition=definition,
                disable_gpu=disable_gpu,
            )

        if "benchmark" in checks:
            if definition is None:
                report.benchmark = make_result(
                    [
                        CheckMessage(
                            level="error", message="cannot benchmark: definition failed to load"
                        )
                    ]
                )
            else:
                baseline_solutions = [
                    e.solution
                    for e in solution_entries
                    if e.author == "baseline" and e.solution is not None
                ]
                report.benchmark = check_benchmark_content(
                    definition_name, definition, workload_traces, baseline_solutions, dataset_root
                )

        report.status = compute_definition_status(report)
        definition_reports[definition_name] = report

    dataset_report = DatasetReport(config=report_config, definitions=definition_reports)

    if "json" in outputs or "text" in outputs:
        report_output_folder.mkdir(parents=True, exist_ok=True)

    if "json" in outputs:
        json_path.write_text(dataset_report.model_dump_json(indent=2))
        logger.info(f"Report JSON written to {json_path}")

    if "text" in outputs or "stdout" in outputs:
        text_output = render_text_report(dataset_report)
        if "text" in outputs:
            text_path.write_text(text_output)
            logger.info(f"Report text written to {text_path}")
        if "stdout" in outputs:
            print(text_output)

    return dataset_report
