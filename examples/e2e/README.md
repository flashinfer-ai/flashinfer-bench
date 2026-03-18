# End-to-End Benchmark Launchers

## What `compare_api_models*.py` measures

`compare_api_models.py` and `compare_api_models_round_traces.py` do **not** benchmark full-model end-to-end latency.

They:

1. Generate candidate kernels
2. Run `flashinfer_bench.bench.Benchmark`
3. Report per-kernel `latency_ms`, `reference_latency_ms`, and `speedup_factor`

Those metrics come from isolated kernel benchmarking on FlashInfer-Trace workloads, not from a real LLM framework request path.

## What `apply()` does

`flashinfer_bench.apply.apply()` is a runtime router:

- It builds an apply table from FlashInfer-Trace benchmark results
- It matches runtime inputs to the best recorded `solution`
- It dispatches to that solution and falls back to the original implementation on a miss
- The winner is chosen from `traces/*.jsonl`, not from `solution.spec.target_hardware`

Current automatic integrations in this repository are:

- `flashinfer.*` wrappers for paged/ragged attention and RMSNorm
- `torch.nn.functional.linear` via the GEMM trace definitions

So real end-to-end gains are only possible if your framework execution actually passes through one of those patched call sites.

## Local environment notes

On this machine:

- `/data1/workspace/airulan/env124.sh` only selects CUDA 12.4 toolchain
- `/data1/workspace/airulan/env130.sh` only selects CUDA 13.0 toolchain
- `/data/workspace/airulan/conda_envs/fib` currently has `torch` and `flashinfer`
- `transformers`, `vllm`, and `sglang` are not installed in `fib`

Recommended pattern:

```bash
conda create --prefix /data/workspace/airulan/conda_envs/fib_e2e \
  --clone /data/workspace/airulan/conda_envs/fib

conda activate /data/workspace/airulan/conda_envs/fib_e2e
source /data1/workspace/airulan/env124.sh

pip install -e /data1/workspace/airulan/bench/flashinfer-bench
pip install transformers accelerate sentencepiece
# Then add whichever stack you want to test:
# pip install vllm
# pip install "sglang[all]"
```

Quick environment check:

```bash
conda activate /data/workspace/airulan/conda_envs/fib_e2e
source /data1/workspace/airulan/env124.sh

python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/check_framework_env.py
```

This prints a JSON report for:

- `torch` and CUDA visibility
- `transformers`
- `vllm`
- `sglang`
- `nvidia-smi`

If `vllm` fails with an `undefined symbol` error, that is usually a binary ABI mismatch between
the installed `vllm` wheel and the current `torch` / CUDA stack. Reinstall `vllm` in the same
environment after you settle the exact `torch` and CUDA toolchain combination.

## 1. Transformers minimal E2E benchmark

This launcher measures real `model.generate()` wall-clock latency:

```bash
conda activate /data/workspace/airulan/conda_envs/fib_e2e
source /data1/workspace/airulan/env124.sh

python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/transformers_generate_benchmark.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --device cuda:0 \
  --dtype bfloat16 \
  --prompt-length 1024 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 5 \
  --enable-apply \
  --apply-scope gemm_only \
  --on-miss-policy fallback_only \
  --aot-ratio 0 \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace


python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/transformers_generate_benchmark.py \
  --model "/data1/hf_models/Meta-Llama-3-8B-Instruct" \
  --device cuda:0 \
  --dtype bfloat16 \
  --prompt-length 1024 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 5 \
  --enable-apply \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace
```

Notes:

- This is a real end-to-end latency measurement for the full `generate()` path
- `--enable-apply` turns on the `torch.nn.functional.linear` patch and any `flashinfer` patches available in-process
- `--apply-scope gemm_only` is the safe default for vanilla Transformers models
- For plain Hugging Face models, gains are usually limited to operators that still go through `F.linear`
- `--solution-pool generated_only` means "only choose from generated kernels"
- `--solution-pool baseline_only` means "only choose from baseline/reference solutions in the trace set"
- `--solution-pool all` means "choose the offline winner from generated + baseline together"
- If you want to force a specific kernel into the model for E2E testing, use `--pin-solution <solution_name>`
- If you want to replace only one definition family, use `--only-definition <definition_name>`
- Pinned solutions are registered with `use_def_best` and restricted to the selected definition(s)
- Pinning only works when the selected solution already has at least one `PASSED` trace under your requested tolerances
- Benchmark JSON now includes:
  - `apply_dispatch_stats`: raw per-definition dispatch counters
  - `apply_replaced_definitions`: only the definitions that actually selected a replacement
  - `apply_selected_solutions`: flattened selected-solution call counts

Example: pin one specific GEMM kernel instead of using the default apply winner:

```bash
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/transformers_generate_benchmark.py \
  --model "/data1/hf_models/Meta-Llama-3-8B-Instruct" \
  --device cuda:0 \
  --dtype float16 \
  --prompt-length 16 \
  --max-new-tokens 1 \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --enable-apply \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --pin-solution gpt-5_gemm_n4096_k4096_cuda_optimized_r4_c0_high

python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/transformers_generate_benchmark.py \
  --model "/data1/hf_models/Meta-Llama-3-8B-Instruct" \
  --device cuda:0 \
  --dtype float16 \
  --prompt-length 16 \
  --max-new-tokens 1 \
  --warmup-runs 1 \
  --benchmark-runs 5 \
  --enable-apply \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --solution-pool generated_only
```

To inspect which definitions a model can hit and which solutions exist:

```bash
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/inspect_transformers_model_coverage.py \
  --model "/data1/hf_models/Meta-Llama-3-8B-Instruct" \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace

python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/inspect_transformers_model_coverage.py \
  --model "/data1/hf_models/Meta-Llama-3-8B-Instruct" \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --trace-hardware-contains A800 \
  --include-solution-stats \
  --summary-only
```

The inspection JSON now includes:

- `best_trace_by_pool.all`: offline global best trace among generated + baseline
- `best_trace_by_pool.generated_only`: offline best generated trace
- `best_trace_by_pool.baseline_only`: offline best baseline trace
- `solution_stats[*].reference_latency_ms_*`: baseline/reference latency stats
- `solution_stats[*].estimated_tflops_*`: GEMM-only estimated TFLOPS derived from `2*M*N*K / latency`

### Compare multiple Transformers modes at once

Use `compare_transformers_apply_modes.py` when you want one command to run a mode matrix such as:

- `torch`
- `baseline_only`
- `generated_only`
- `pin_solution`

The compare script writes:

- `manifest.json`
- `summary.csv`
- `summary.json`
- `runs/<mode>/result.json`
- `runs/<mode>/stdout.log`
- `runs/<mode>/stderr.log`

Example:

```bash
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/compare_transformers_apply_modes.py \
  --output-dir /tmp/fib_transformers_compare \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --trace-hardware-contains A800 \
  --pin-solution gpt-5_gemm_n4096_k4096_cuda_optimized_r4_c0_high \
  -- \
  --model /data1/hf_models/Meta-Llama-3-8B-Instruct \
  --device cuda:0 \
  --dtype float16 \
  --prompt-length 16 \
  --max-new-tokens 1 \
  --warmup-runs 0 \
  --benchmark-runs 1
```

If you also want the offline global-best mode, add `--modes torch baseline_only generated_only pin_solution all`.

## 2. Generic bootstrap launcher for vLLM / SGLang

Use `bootstrap_apply_runner.py` when you want `apply` enabled **before** the framework imports its kernels and then let the framework's own benchmark command run normally.

Important:

- For offline benchmarks, wrap the benchmark process itself
- For online serving benchmarks, wrap the **server process**, not just the client benchmark

### vLLM example

Offline latency benchmark in a single process:

```bash
conda activate /data/workspace/airulan/conda_envs/fib_e2e
source /data1/workspace/airulan/env124.sh

python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/bootstrap_apply_runner.py \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --on-miss-policy fallback_only \
  --aot-ratio 0 \
  --script /data/workspace/airulan/conda_envs/fib_e2e/bin/vllm \
  -- bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 1024 \
  --output-len 128
```

OpenAI-compatible server with apply enabled, then benchmark it with vLLM's own client benchmark:

```bash
# Terminal 1: launch the server with apply enabled
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/bootstrap_apply_runner.py \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --on-miss-policy fallback_only \
  --aot-ratio 0 \
  --script /data/workspace/airulan/conda_envs/fib_e2e/bin/vllm \
  -- serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Terminal 2: benchmark the running server
/data/workspace/airulan/conda_envs/fib_e2e/bin/vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 100
```

### SGLang example

Offline single-batch benchmark:

```bash
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/bootstrap_apply_runner.py \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --on-miss-policy fallback_only \
  --aot-ratio 0 \
  --module sglang.bench_one_batch \
  -- --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --batch 1 \
  --input-len 1024 \
  --output-len 128
```

Online serving benchmark after you launch a server with apply enabled:

```bash
# Terminal 1: launch the server with apply enabled
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/bootstrap_apply_runner.py \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --on-miss-policy fallback_only \
  --aot-ratio 0 \
  --module sglang.launch_server \
  -- --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 30000

# Terminal 2: benchmark the running server
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 32 \
  --random-input-len 1024 \
  --random-output-len 128
```

Why this launcher exists:

- `enable_apply()` is process-local
- The monkey patches must be installed before the framework starts caching/importing the target functions
- Running the target inside the same Python process guarantees the patches stay active
- `--pin-solution` and `--only-definition` can be used here too when you want to isolate one kernel family in vLLM or SGLang
- `--solution-pool generated_only` can be used here too when you want the framework to pick only generated kernels

### Compare multiple bootstrap modes at once

Use `compare_bootstrap_apply_modes.py` when you want to run the same **single-process** module/script under:

- `torch`
- `baseline_only`
- `generated_only`
- `pin_solution`

The compare driver launches each mode separately, captures stdout/stderr, and optionally parses a
framework-native JSON/JSONL artifact if you tell it where that artifact will be written.

This is the right tool for:

- `transformers_generate_benchmark.py`
- `vllm bench latency`
- `sglang.bench_one_batch`

This is **not** the right tool for online serving client benchmarks such as `vllm bench serve` or
`python -m sglang.bench_serving`, because those only measure an already-running server. For online
serving, use `compare_server_apply_modes.py` below so that each mode launches its own server.

Supported placeholders for target arguments and `--target-result-path`:

- `{mode}`
- `{mode_slug}`
- `{output_dir}`
- `{mode_output_dir}`

SGLang offline one-batch example:

```bash
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/compare_bootstrap_apply_modes.py \
  --output-dir /tmp/fib_sglang_one_batch_compare \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --trace-hardware-contains A800 \
  --pin-solution gpt-5_gemm_n4096_k4096_cuda_optimized_r4_c0_high \
  --module sglang.bench_one_batch \
  --target-result-path {mode_output_dir}/bench_one_batch.jsonl \
  --target-result-format jsonl \
  -- \
  --model-path /data1/hf_models/Meta-Llama-3-8B-Instruct \
  --batch-size 1 \
  --input-len 1024 \
  --output-len 128 \
  --result-filename {mode_output_dir}/bench_one_batch.jsonl
```

vLLM offline benchmark example:

```bash
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/compare_bootstrap_apply_modes.py \
  --output-dir /tmp/fib_vllm_compare \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --trace-hardware-contains A800 \
  --pin-solution gpt-5_gemm_n4096_k4096_cuda_optimized_r4_c0_high \
  --script /data/workspace/airulan/conda_envs/fib_e2e/bin/vllm \
  --target-result-path {mode_output_dir}/latency.json \
  --target-result-format json \
  -- \
  bench latency \
  --model /data1/hf_models/Meta-Llama-3-8B-Instruct \
  --input-len 1024 \
  --output-len 128 \
  --output-json {mode_output_dir}/latency.json
```

### Compare server/client serving modes at once

Use `compare_server_apply_modes.py` when the benchmark has two processes:

- a server process where `apply` must be enabled
- a client benchmark process that talks to the server

The script launches one server per mode, waits for readiness, runs the client benchmark, collects:

- `manifest.json`
- `summary.csv`
- `summary.json`
- `runs/<mode>/server_stdout.log`
- `runs/<mode>/server_stderr.log`
- `runs/<mode>/client_stdout.log`
- `runs/<mode>/client_stderr.log`
- `runs/<mode>/apply_summary.json` for apply-enabled modes

vLLM online serving example:

```bash
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/compare_server_apply_modes.py \
  --output-dir /tmp/fib_vllm_serve_compare \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --trace-hardware-contains A800 \
  --pin-solution gpt-5_gemm_n4096_k4096_cuda_optimized_r4_c0_high \
  --server-script /data/workspace/airulan/conda_envs/fib_e2e/bin/vllm \
  --ready-url http://127.0.0.1:8000/v1/models \
  --client-command "/data/workspace/airulan/conda_envs/fib_e2e/bin/vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model /data1/hf_models/Meta-Llama-3-8B-Instruct --dataset-name random --num-prompts 32 --random-input-len 1024 --random-output-len 128 --save-result --result-dir {mode_output_dir} --result-filename serve.json" \
  --client-result-path {mode_output_dir}/serve.json \
  --client-result-format json \
  -- \
  serve /data1/hf_models/Meta-Llama-3-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000
```

SGLang online serving example:

```bash
python /data1/workspace/airulan/bench/flashinfer-bench/examples/e2e/compare_server_apply_modes.py \
  --output-dir /tmp/fib_sglang_serve_compare \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --trace-hardware-contains A800 \
  --pin-solution gpt-5_gemm_n4096_k4096_cuda_optimized_r4_c0_high \
  --server-module sglang.launch_server \
  --ready-url http://127.0.0.1:30000/v1/models \
  --client-command "/data/workspace/airulan/conda_envs/fib_e2e/bin/python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --model /data1/hf_models/Meta-Llama-3-8B-Instruct --dataset-name random --num-prompts 32 --random-input-len 1024 --random-output-len 128 --output-file {mode_output_dir}/serving.jsonl" \
  --client-result-path {mode_output_dir}/serving.jsonl \
  --client-result-format jsonl \
  -- \
  --model-path /data1/hf_models/Meta-Llama-3-8B-Instruct \
  --host 127.0.0.1 \
  --port 30000
```

## Interpreting results

For real E2E benchmarking you should compare:

1. Framework benchmark without apply
2. Same benchmark with apply enabled
3. Same model, same prompt lengths, same batch size, same CUDA/toolchain env

If the framework does not touch a patched operator, E2E latency will not move even if kernel-level benchmark numbers look better.
