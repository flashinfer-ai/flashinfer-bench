#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="/data/workspace/airulan/conda_envs/fib/bin/python"
CUDA_ENV_SH="/data1/workspace/airulan/env124.sh"
CLI_BIN=""

# Full external trace set with definitions/workloads/blob/traces/solutions.
TRACE_SET_ROOT="/data1/workspace/airulan/bench/flashinfer-trace"
# Bundled repo copy with lightweight definitions/workloads examples only.
BUNDLED_TRACE_ROOT="/data1/workspace/airulan/bench/flashinfer-bench/flashinfer_trace"

SOLUTION_DIRS=(
  "/data1/workspace/airulan/bench/test/gpt-5.2_n"
  #"/data1/workspace/airulan/bench/test/Qwen3.5-27B"
)
SOLUTION_DIRS_DEFAULTED="1"

OUTPUT_BASE="/tmp/fib_solution_dir_benchmarks_$(date +%Y%m%d_%H%M%S)"
WARMUP_RUNS="10"
ITERATIONS="50"
NUM_TRIALS="3"
TIMEOUT_SECONDS="300"
LOG_LEVEL="INFO"
USE_ISOLATED_RUNNER="0"
RESUME="0"
REQUIRED_MATCHED_RATIO=""
DEFINITIONS=()
SOLUTIONS=()

usage() {
  cat <<'EOF'
Usage:
  run_solution_dir_benchmarks.sh [options]

What this script does:
  1. Uses the full external trace set at /data1/workspace/airulan/bench/flashinfer-trace
  2. Builds a temporary TraceSet per solution author under /tmp by symlinking:
       - definitions/
       - workloads/
       - blob/
  3. Copies one or more solution-author directories into solutions/
  4. Runs `flashinfer_bench run`
  5. Writes benchmark traces plus text/JSON summaries

Options:
  --python PATH
  --cuda-env-sh PATH
  --trace-set-root PATH
  --output-base PATH
  --solution-dir PATH
  --warmup-runs INT
  --iterations INT
  --num-trials INT
  --timeout INT
  --log-level LEVEL
  --use-isolated-runner
  --resume
  --required-matched-ratio FLOAT
  --definitions NAME [NAME ...]
  --solutions NAME [NAME ...]
  --help

Examples:
  # Benchmark both default author directories
  run_solution_dir_benchmarks.sh

  # Benchmark only one definition for gpt-5.2
  run_solution_dir_benchmarks.sh \
    --solution-dir /data1/workspace/airulan/bench/test/gpt-5.2 \
    --definitions fused_add_rmsnorm_h4096
EOF
}

while (($#)); do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --cuda-env-sh)
      CUDA_ENV_SH="$2"
      shift 2
      ;;
    --trace-set-root)
      TRACE_SET_ROOT="$2"
      shift 2
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --solution-dir)
      if [[ "${SOLUTION_DIRS_DEFAULTED}" == "1" ]]; then
        SOLUTION_DIRS=()
        SOLUTION_DIRS_DEFAULTED="0"
      fi
      SOLUTION_DIRS+=("$2")
      shift 2
      ;;
    --warmup-runs)
      WARMUP_RUNS="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --num-trials)
      NUM_TRIALS="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --use-isolated-runner)
      USE_ISOLATED_RUNNER="1"
      shift
      ;;
    --resume)
      RESUME="1"
      shift
      ;;
    --required-matched-ratio)
      REQUIRED_MATCHED_RATIO="$2"
      shift 2
      ;;
    --definitions)
      shift
      DEFINITIONS=()
      while (($#)) && [[ "$1" != --* ]]; do
        DEFINITIONS+=("$1")
        shift
      done
      ;;
    --solutions)
      shift
      SOLUTIONS=()
      while (($#)) && [[ "$1" != --* ]]; do
        SOLUTIONS+=("$1")
        shift
      done
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

note() {
  printf '[bench] %s\n' "$*"
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory not found: $1" >&2
    exit 1
  fi
}

resolve_cli_launcher() {
  local candidate=""
  candidate="$(dirname "${PYTHON_BIN}")/flashinfer-bench"
  if [[ -x "${candidate}" ]]; then
    CLI_BIN="${candidate}"
    return
  fi
  CLI_BIN=""
}

link_into_root() {
  local target="$1"
  local link_path="$2"
  if [[ -e "${link_path}" || -L "${link_path}" ]]; then
    echo "Refusing to overwrite existing path: ${link_path}" >&2
    exit 1
  fi
  ln -s "${target}" "${link_path}"
}

build_reports() {
  local eval_root="$1"
  local summary_txt="$2"
  local best_txt="$3"
  local output_json="$4"
  local trace_records_jsonl="$5"
  "${PYTHON_BIN}" - "${eval_root}" "${summary_txt}" "${best_txt}" "${output_json}" "${trace_records_jsonl}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from flashinfer_bench.data import TraceSet

eval_root = Path(sys.argv[1])
summary_txt = Path(sys.argv[2])
best_txt = Path(sys.argv[3])
output_json = Path(sys.argv[4])
trace_records_jsonl = Path(sys.argv[5])
trace_set = TraceSet.from_path(str(eval_root))

all_traces = [trace for traces in trace_set.traces.values() for trace in traces]
status_counts = Counter(
    trace.evaluation.status.value if trace.evaluation is not None else "MISSING"
    for trace in all_traces
)
evaluated_definitions = sorted(trace_set.traces)

trace_record_lines = []
for def_name in evaluated_definitions:
    for trace in trace_set.traces.get(def_name, []):
        evaluation = trace.evaluation
        correctness = evaluation.correctness if evaluation is not None else None
        performance = evaluation.performance if evaluation is not None else None
        trace_record_lines.append(
            json.dumps(
                {
                    "definition": def_name,
                    "solution": trace.solution,
                    "workload_uuid": trace.workload.uuid,
                    "axes": dict(trace.workload.axes),
                    "status": evaluation.status.value if evaluation is not None else "MISSING",
                    "latency_ms": performance.latency_ms if performance is not None else None,
                    "reference_latency_ms": (
                        performance.reference_latency_ms if performance is not None else None
                    ),
                    "speedup_factor": performance.speedup_factor if performance is not None else None,
                    "max_absolute_error": (
                        correctness.max_absolute_error if correctness is not None else None
                    ),
                    "max_relative_error": (
                        correctness.max_relative_error if correctness is not None else None
                    ),
                },
                ensure_ascii=False,
            )
        )
trace_records_jsonl.write_text("\n".join(trace_record_lines) + ("\n" if trace_record_lines else ""), encoding="utf-8")

per_definition = {}
for def_name in evaluated_definitions:
    traces = trace_set.traces.get(def_name, [])
    if not traces:
        continue
    def_status_counts = Counter(
        trace.evaluation.status.value if trace.evaluation is not None else "MISSING"
        for trace in traces
    )
    best = trace_set.get_best_trace(def_name)
    per_definition[def_name] = {
        "total": len(traces),
        "passed": def_status_counts.get("PASSED", 0),
        "failed": len(traces) - def_status_counts.get("PASSED", 0),
        "status_counts": dict(sorted(def_status_counts.items())),
        "best_solution": best.solution if best is not None else None,
        "best_speedup_factor": (
            best.evaluation.performance.speedup_factor
            if best is not None and best.evaluation is not None and best.evaluation.performance is not None
            else None
        ),
        "best_latency_ms": (
            best.evaluation.performance.latency_ms
            if best is not None and best.evaluation is not None and best.evaluation.performance is not None
            else None
        ),
        "best_reference_latency_ms": (
            best.evaluation.performance.reference_latency_ms
            if best is not None and best.evaluation is not None and best.evaluation.performance is not None
            else None
        ),
        "best_max_absolute_error": (
            best.evaluation.correctness.max_absolute_error
            if best is not None and best.evaluation is not None and best.evaluation.correctness is not None
            else None
        ),
        "best_max_relative_error": (
            best.evaluation.correctness.max_relative_error
            if best is not None and best.evaluation is not None and best.evaluation.correctness is not None
            else None
        ),
    }

payload = {
    "trace_set_root": str(eval_root),
    "summary": trace_set.summary(),
    "status_counts": dict(sorted(status_counts.items())),
    "per_definition": per_definition,
}
output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

summary_payload = trace_set.summary()
summary_lines = [
    f"TraceSet root: {eval_root}",
    f"Total traces: {summary_payload['total']}",
    f"Passed traces: {summary_payload['passed']}",
    f"Failed traces: {summary_payload['failed']}",
    f"Min latency (passed): {summary_payload['min_latency_ms']}",
    f"Max latency (passed): {summary_payload['max_latency_ms']}",
    f"Avg latency (passed): {summary_payload['avg_latency_ms']}",
    "",
    "Status counts:",
]
for status, count in sorted(status_counts.items()):
    summary_lines.append(f"  {status}: {count}")
summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

best_lines = [f"TraceSet root: {eval_root}", ""]
for def_name in sorted(trace_set.definitions):
    best = trace_set.get_best_trace(def_name)
    if best is None:
        best_lines.append(f"{def_name}: no passing trace")
        continue
    perf = best.evaluation.performance
    corr = best.evaluation.correctness
    best_lines.append(f"{def_name}:")
    best_lines.append(f"  solution: {best.solution}")
    best_lines.append(f"  speedup_factor: {perf.speedup_factor}")
    best_lines.append(f"  latency_ms: {perf.latency_ms}")
    best_lines.append(f"  reference_latency_ms: {perf.reference_latency_ms}")
    best_lines.append(f"  max_absolute_error: {corr.max_absolute_error}")
    best_lines.append(f"  max_relative_error: {corr.max_relative_error}")
best_txt.write_text("\n".join(best_lines) + "\n", encoding="utf-8")
PY
}

if [[ -n "${CUDA_ENV_SH}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${CUDA_ENV_SH}"
  set -u
fi

require_dir "${REPO_ROOT}"
require_dir "${TRACE_SET_ROOT}"
require_dir "${TRACE_SET_ROOT}/definitions"
require_dir "${TRACE_SET_ROOT}/workloads"
require_dir "${TRACE_SET_ROOT}/blob"
resolve_cli_launcher

mkdir -p "${OUTPUT_BASE}"

note "Using full trace set: ${TRACE_SET_ROOT}"
note "Bundled repo trace directory (reference only): ${BUNDLED_TRACE_ROOT}"
note "Output base: ${OUTPUT_BASE}"
if [[ -n "${CLI_BIN}" ]]; then
  note "Using CLI entrypoint: ${CLI_BIN}"
else
  note "Using CLI entrypoint via Python fallback"
fi

cd "${REPO_ROOT}"

for solution_dir in "${SOLUTION_DIRS[@]}"; do
  require_dir "${solution_dir}"
  author="$(basename "${solution_dir}")"
  eval_root="${OUTPUT_BASE}/${author}"

  if [[ -e "${eval_root}" ]]; then
    echo "Output directory already exists: ${eval_root}" >&2
    exit 1
  fi

  note "Preparing evaluation trace set for author: ${author}"

  mkdir -p "${eval_root}/solutions" "${eval_root}/traces"
  link_into_root "${TRACE_SET_ROOT}/definitions" "${eval_root}/definitions"
  link_into_root "${TRACE_SET_ROOT}/workloads" "${eval_root}/workloads"
  link_into_root "${TRACE_SET_ROOT}/blob" "${eval_root}/blob"

  cp -a "${solution_dir}" "${eval_root}/solutions/${author}"

  if [[ -n "${CLI_BIN}" ]]; then
    cmd=(
      "${CLI_BIN}"
      run
      --local "${eval_root}"
      --save-results
      --log-level "${LOG_LEVEL}"
      --warmup-runs "${WARMUP_RUNS}"
      --iterations "${ITERATIONS}"
      --num-trials "${NUM_TRIALS}"
      --timeout "${TIMEOUT_SECONDS}"
    )
  else
    cmd=(
      "${PYTHON_BIN}"
      -c "from flashinfer_bench.cli.main import cli; cli()"
      run
      --local "${eval_root}"
      --save-results
      --log-level "${LOG_LEVEL}"
      --warmup-runs "${WARMUP_RUNS}"
      --iterations "${ITERATIONS}"
      --num-trials "${NUM_TRIALS}"
      --timeout "${TIMEOUT_SECONDS}"
    )
  fi

  if [[ "${USE_ISOLATED_RUNNER}" == "1" ]]; then
    cmd+=(--use-isolated-runner)
  fi
  if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
  fi
  if [[ -n "${REQUIRED_MATCHED_RATIO}" ]]; then
    cmd+=(--required-matched-ratio "${REQUIRED_MATCHED_RATIO}")
  fi
  if ((${#DEFINITIONS[@]})); then
    cmd+=(--definitions "${DEFINITIONS[@]}")
  fi
  if ((${#SOLUTIONS[@]})); then
    cmd+=(--solutions "${SOLUTIONS[@]}")
  fi

  note "Running benchmark for ${author}"
  {
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
  } 2>&1 | tee "${eval_root}/run.log"

  note "Writing reports for ${author}"
  build_reports \
    "${eval_root}" \
    "${eval_root}/summary.txt" \
    "${eval_root}/best.txt" \
    "${eval_root}/artifact_summary.json" \
    "${eval_root}/trace_records.jsonl"

  note "Completed ${author}. Results stored in ${eval_root}"
done

note "All benchmarks finished."
