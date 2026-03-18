#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="/data/workspace/airulan/conda_envs/fib_e2e/bin/python"
CUDA_ENV_SH="/data1/workspace/airulan/env124.sh"
MODEL_PATH="/data1/hf_models/Llama-3.1-8B"
OUTPUT_DIR="/tmp/sglang_rmsnorm_repro"
TRACE_SET_PATH="/data1/workspace/airulan/bench/flashinfer-trace"
DEFINITION="fused_add_rmsnorm_h4096"
SOLUTION_NAME="gpt-5_fused_add_rmsnorm_h4096_cuda_optimized_r4_c0_high"
HIDDEN_SIZE="4096"
DEVICE="cuda:0"
DTYPE="bfloat16"
INPUT_LEN="16"
OUTPUT_LEN="1"
TRACE_HARDWARE_CONTAINS="A800"
ATTENTION_BACKEND="flashinfer"
SAMPLING_BACKEND="pytorch"
MAX_ATOL="0.05"
MAX_RTOL="0.01"
FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_ws_fib_e2e"
FIB_CACHE_PATH=""
BATCH_SIZES=("1" "16" "64")

usage() {
  cat <<'EOF'
Usage:
  run_sglang_rmsnorm_repro.sh [options]

Options:
  --python PATH
  --cuda-env-sh PATH
  --model-path PATH
  --output-dir PATH
  --trace-set-path PATH
  --definition NAME
  --solution-name NAME
  --hidden-size INT
  --device DEV
  --dtype {bfloat16,float16}
  --input-len INT
  --output-len INT
  --batch-sizes N [N...]
  --trace-hardware-contains TOKEN
  --attention-backend NAME
  --sampling-backend NAME
  --max-atol FLOAT
  --max-rtol FLOAT
  --flashinfer-workspace-base PATH
  --fib-cache-path PATH
  --help

Example:
  run_sglang_rmsnorm_repro.sh \
    --model-path /data1/hf_models/Llama-3.1-8B \
    --output-dir /tmp/sglang_rmsnorm_repro_full
EOF
}

while (($#)); do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --cuda-env-sh) CUDA_ENV_SH="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --trace-set-path) TRACE_SET_PATH="$2"; shift 2 ;;
    --definition) DEFINITION="$2"; shift 2 ;;
    --solution-name) SOLUTION_NAME="$2"; shift 2 ;;
    --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --input-len) INPUT_LEN="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --trace-hardware-contains) TRACE_HARDWARE_CONTAINS="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --sampling-backend) SAMPLING_BACKEND="$2"; shift 2 ;;
    --max-atol) MAX_ATOL="$2"; shift 2 ;;
    --max-rtol) MAX_RTOL="$2"; shift 2 ;;
    --flashinfer-workspace-base) FLASHINFER_WORKSPACE_BASE="$2"; shift 2 ;;
    --fib-cache-path) FIB_CACHE_PATH="$2"; shift 2 ;;
    --batch-sizes)
      shift
      BATCH_SIZES=()
      while (($#)) && [[ "$1" != --* ]]; do
        BATCH_SIZES+=("$1")
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

if [[ -n "${CUDA_ENV_SH}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${CUDA_ENV_SH}"
  set -u
fi

export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE}"
mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/reproduce_sglang_rmsnorm_experiment.py"
  --output-dir "${OUTPUT_DIR}"
  --model-path "${MODEL_PATH}"
  --trace-set-path "${TRACE_SET_PATH}"
  --definition "${DEFINITION}"
  --solution-name "${SOLUTION_NAME}"
  --hidden-size "${HIDDEN_SIZE}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --input-len "${INPUT_LEN}"
  --output-len "${OUTPUT_LEN}"
  --trace-hardware-contains "${TRACE_HARDWARE_CONTAINS}"
  --attention-backend "${ATTENTION_BACKEND}"
  --sampling-backend "${SAMPLING_BACKEND}"
  --max-atol "${MAX_ATOL}"
  --max-rtol "${MAX_RTOL}"
  --flashinfer-workspace-base "${FLASHINFER_WORKSPACE_BASE}"
  --batch-sizes "${BATCH_SIZES[@]}"
)

if [[ -n "${FIB_CACHE_PATH}" ]]; then
  CMD+=(--fib-cache-path "${FIB_CACHE_PATH}")
fi

printf 'Running command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
