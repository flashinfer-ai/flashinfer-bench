#!/usr/bin/env bash
set -uo pipefail

export LLM_API_KEY="sk-FxxOqLn6bLB0VP3NAo7xmRK39lmHKlma90HYjomAje5jgatP"
source "/data1/workspace/airulan/env124.sh"

RUN_DIR="/data1/workspace/airulan/bench/flashinfer-bench/logs/fib_batch_gpt5_cuda_20260312_205038"
STATUS_FILE="${RUN_DIR}/status.tsv"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
RUNNER_LOG="${RUN_DIR}/runner.log"

mkdir -p "${RUN_DIR}"

echo -e "definition\tattempt\tresult\texit_code\tlog_file" > "${STATUS_FILE}"
echo "Batch run started at: $(date)" > "${SUMMARY_FILE}"
echo >> "${SUMMARY_FILE}"

run_one_definition() {
  local def="$1"
  local attempt=1

  while [ $attempt -le 2 ]; do
    local log_file="${RUN_DIR}/${def}_attempt${attempt}.log"

    echo "==================================================" | tee -a "${RUNNER_LOG}"
    echo "[$(date)] Start definition: ${def} | attempt ${attempt}/2" | tee -a "${RUNNER_LOG}"
    echo "Log: ${log_file}" | tee -a "${RUNNER_LOG}"
    echo "==================================================" | tee -a "${RUNNER_LOG}"

    "/data/workspace/airulan/conda_envs/fib/bin/python" "/data1/workspace/airulan/bench/flashinfer-bench/examples/kernel_generator/compare_api_models_round_traces.py"       --trace-set-path "/data1/workspace/airulan/bench/flashinfer-trace"       --base-url "https://aigc.x-see.cn/v1"       --models "gpt-5"       --definitions "${def}"       --language "cuda"       --target-gpu "A800"       --gen-rounds 10       --feedback-warmup-runs 2       --feedback-iterations 10       --feedback-num-trials 1       --warmup-runs 5       --iterations 20       --num-trials 3       --timeout 300       --fail-fast       2>&1 | tee "${log_file}"

    exit_code=${PIPESTATUS[0]}

    if [ "${exit_code}" -eq 0 ]; then
      echo -e "${def}\t${attempt}\tSUCCESS\t${exit_code}\t${log_file}" >> "${STATUS_FILE}"
      echo "[SUCCESS] ${def} succeeded on attempt ${attempt}" | tee -a "${RUNNER_LOG}"
      echo "SUCCESS  ${def}  attempt=${attempt}" >> "${SUMMARY_FILE}"
      return 0
    else
      echo -e "${def}\t${attempt}\tFAIL\t${exit_code}\t${log_file}" >> "${STATUS_FILE}"
      echo "[FAIL] ${def} failed on attempt ${attempt} with exit code ${exit_code}" | tee -a "${RUNNER_LOG}"

      if [ $attempt -lt 2 ]; then
        echo "[RETRY] Retrying ${def}..." | tee -a "${RUNNER_LOG}"
        attempt=$((attempt + 1))
      else
        echo "[SKIP] ${def} failed twice, skip to next definition." | tee -a "${RUNNER_LOG}"
        echo "FAILED   ${def}  attempts=2" >> "${SUMMARY_FILE}"
        return 1
      fi
    fi
  done
}

definitions=(
  "gemm_n128_k2048"
  "gemm_n2048_k4096"
  "rmsnorm_h5120"
  "gemm_n5120_k5120"
)

for def in "${definitions[@]}"; do
  run_one_definition "${def}"
done

echo >> "${SUMMARY_FILE}"
echo "Batch run finished at: $(date)" >> "${SUMMARY_FILE}"

echo
echo "All definitions processed."
echo "Run dir: ${RUN_DIR}"
echo "Status file: ${STATUS_FILE}"
echo "Summary file: ${SUMMARY_FILE}"

exec bash
