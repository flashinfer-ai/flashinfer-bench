# Kernel Generator

A multi-turn kernel generating agent that uses FlashInfer-Bench for evaluation feedback. It can conduct sequential multi-turn generation and beam search kernel exploration.

## Usage

1. **Configure generation settings** in `kernel_generator_example.py`:
   - Set `model_name` (e.g., `"gpt-5-2025-08-07"`)
   - Set `language` (`"cuda"` or `"triton"`, will support more in the future)
   - Set `target_gpu` (e.g., `"B200"`, `"H100"`, `"A100"`)
   - Optionally set `definition` to target a specific kernel (leave empty to generate all definitions in the traceset)

2. **Set traceset path**:
   - Update `traceset_path` to your flashinfer-trace dataset directory

3. **To Enable beam search**:
   - Uncomment lines 97-98 to use beam search mode

4. **Set API credentials**:
   - Create a `.env` file by following the .env.example:
     ```
     LLM_API_KEY=your_api_key
     BASE_URL=your_base_url  # Optional, for non-OpenAI APIs
     ```

5. **Run the generator**:
   ```bash
   python kernel_generator_example.py
   ```

Generated solutions are saved to `{traceset_path}/solutions/{author}/{op_type}/{definition_name}/{solution_name}.json`

## Batch API Comparison

For multi-model CUDA generation experiments against an OpenAI-compatible proxy such as `https://aigc.x-see.cn/v1`, use `compare_api_models.py`.

### What it does

1. Discovers models from `/models` or uses an explicit `--models` list
2. Generates kernels with a fixed feedback workload per definition for fair comparison
3. Runs the final FlashInfer-Bench benchmark on all workloads of that definition
4. Writes CSV summaries for trace-level results and finer-grained error buckets

### Evaluation Standard

The final benchmark still uses the native FlashInfer-Bench schema:

- `COMPILE_ERROR`
- `RUNTIME_ERROR`
- `INCORRECT_SHAPE`
- `INCORRECT_DTYPE`
- `INCORRECT_NUMERICAL`
- `TIMEOUT`
- `PASSED`

For `PASSED`, FlashInfer-Bench reports:

- `max_absolute_error`
- `max_relative_error`
- `latency_ms`
- `reference_latency_ms`
- `speedup_factor`

The new comparison script adds a secondary taxonomy on top of that, for example:

- `compile.signature_mismatch`
- `compile.cuda_compile`
- `runtime.cuda_launch`
- `runtime.out_of_memory`
- `correctness.shape`
- `correctness.nonfinite`
- `efficiency.regression`
- `efficiency.breakout`

### Example

List the models exposed by the proxy:

```bash
/data/workspace/airulan/conda_envs/fib/bin/python compare_api_models.py \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --list-models
```

Compare the latest matching model for each prefix:

```bash
export LLM_API_KEY=your_api_key

/data/workspace/airulan/conda_envs/fib/bin/python compare_api_models.py \
  --trace-set-path /data1/workspace/airulan/bench/flashinfer-trace \
  --base-url https://aigc.x-see.cn/v1 \
  --model-prefixes claude gpt-5 o3 \
  --definitions rmsnorm_h128 \
  --language cuda \
  --target-gpu H100
```

If `/models` returns creation timestamps, the script uses them to pick the latest match for each prefix. Otherwise it falls back to natural sorting on the model ID. Use `--all-matching-models` to benchmark every discovered match instead of only the latest one.

### Outputs

Each run creates a timestamped directory under `examples/kernel_generator/results/` with:

- `manifest.json`: run configuration
- `available_models.json`: discovered provider models, when applicable
- `experiment_summary.csv`: one row per model-definition experiment
- `trace_records.csv`: one row per benchmark trace
- `error_summary.csv`: grouped counts by fine-grained error bucket
