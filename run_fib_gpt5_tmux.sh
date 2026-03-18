#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="fib_batch_gpt5_cuda"
LOG_DIR="/data1/workspace/airulan/bench/flashinfer-bench/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_DIR}/${SESSION_NAME}_${TIMESTAMP}"
RUNNER_SCRIPT="${RUN_DIR}/_tmux_runner.sh"

mkdir -p "${RUN_DIR}"

# ========== 必填 ==========
export LLM_API_KEY="sk-FxxOqLn6bLB0VP3NAo7xmRK39lmHKlma90HYjomAje5jgatP"

# 把你所有要测的 definitions 填这里
DEFINITIONS=(
  "gemm_n128_k2048"
  "gemm_n2048_k4096"
  "rmsnorm_h5120"
  "gemm_n5120_k5120"
)

# ========== 固定参数 ==========
BASE_URL="https://aigc.x-see.cn/v1"
MODEL="gpt-5"
LANGUAGE="cuda"
TARGET_GPU="A800"
TRACE_SET_PATH="/data1/workspace/airulan/bench/flashinfer-trace"
PYTHON_BIN="/data/workspace/airulan/conda_envs/fib/bin/python"
SCRIPT_PATH="/data1/workspace/airulan/bench/flashinfer-bench/examples/kernel_generator/compare_api_models_round_traces.py"
ENV_SCRIPT="/data1/workspace/airulan/env124.sh"
MAX_RETRIES=2

# 生成 tmux 内真正执行的 runner 脚本
cat > "${RUNNER_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -uo pipefail

export LLM_API_KEY="${LLM_API_KEY}"
source "${ENV_SCRIPT}"

RUN_DIR="${RUN_DIR}"
STATUS_FILE="\${RUN_DIR}/status.tsv"
SUMMARY_FILE="\${RUN_DIR}/summary.txt"
RUNNER_LOG="\${RUN_DIR}/runner.log"

mkdir -p "\${RUN_DIR}"

echo -e "definition\tattempt\tresult\texit_code\tlog_file" > "\${STATUS_FILE}"
echo "Batch run started at: \$(date)" > "\${SUMMARY_FILE}"
echo >> "\${SUMMARY_FILE}"

run_one_definition() {
  local def="\$1"
  local attempt=1

  while [ \$attempt -le ${MAX_RETRIES} ]; do
    local log_file="\${RUN_DIR}/\${def}_attempt\${attempt}.log"

    echo "==================================================" | tee -a "\${RUNNER_LOG}"
    echo "[\$(date)] Start definition: \${def} | attempt \${attempt}/${MAX_RETRIES}" | tee -a "\${RUNNER_LOG}"
    echo "Log: \${log_file}" | tee -a "\${RUNNER_LOG}"
    echo "==================================================" | tee -a "\${RUNNER_LOG}"

    "${PYTHON_BIN}" "${SCRIPT_PATH}" \
      --trace-set-path "${TRACE_SET_PATH}" \
      --base-url "${BASE_URL}" \
      --models "${MODEL}" \
      --definitions "\${def}" \
      --language "${LANGUAGE}" \
      --target-gpu "${TARGET_GPU}" \
      --gen-rounds 10 \
      --feedback-warmup-runs 2 \
      --feedback-iterations 10 \
      --feedback-num-trials 1 \
      --warmup-runs 5 \
      --iterations 20 \
      --num-trials 3 \
      --timeout 300 \
      --fail-fast \
      2>&1 | tee "\${log_file}"

    exit_code=\${PIPESTATUS[0]}

    if [ "\${exit_code}" -eq 0 ]; then
      echo -e "\${def}\t\${attempt}\tSUCCESS\t\${exit_code}\t\${log_file}" >> "\${STATUS_FILE}"
      echo "[SUCCESS] \${def} succeeded on attempt \${attempt}" | tee -a "\${RUNNER_LOG}"
      echo "SUCCESS  \${def}  attempt=\${attempt}" >> "\${SUMMARY_FILE}"
      return 0
    else
      echo -e "\${def}\t\${attempt}\tFAIL\t\${exit_code}\t\${log_file}" >> "\${STATUS_FILE}"
      echo "[FAIL] \${def} failed on attempt \${attempt} with exit code \${exit_code}" | tee -a "\${RUNNER_LOG}"

      if [ \$attempt -lt ${MAX_RETRIES} ]; then
        echo "[RETRY] Retrying \${def}..." | tee -a "\${RUNNER_LOG}"
        attempt=\$((attempt + 1))
      else
        echo "[SKIP] \${def} failed twice, skip to next definition." | tee -a "\${RUNNER_LOG}"
        echo "FAILED   \${def}  attempts=${MAX_RETRIES}" >> "\${SUMMARY_FILE}"
        return 1
      fi
    fi
  done
}

definitions=(
$(printf '  "%s"\n' "${DEFINITIONS[@]}")
)

for def in "\${definitions[@]}"; do
  run_one_definition "\${def}"
done

echo >> "\${SUMMARY_FILE}"
echo "Batch run finished at: \$(date)" >> "\${SUMMARY_FILE}"

echo
echo "All definitions processed."
echo "Run dir: \${RUN_DIR}"
echo "Status file: \${STATUS_FILE}"
echo "Summary file: \${SUMMARY_FILE}"

exec bash
EOF

chmod +x "${RUNNER_SCRIPT}"

# 如果同名 session 已存在，先删掉
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

tmux new-session -d -s "${SESSION_NAME}" "bash '${RUNNER_SCRIPT}'"

echo "tmux session 已启动: ${SESSION_NAME}"
echo "进入会话: tmux attach -t ${SESSION_NAME}"
echo "日志目录: ${RUN_DIR}"
echo "runner 脚本: ${RUNNER_SCRIPT}"