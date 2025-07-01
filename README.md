# flashinfer-bench

## flashinfer-bench leaderboard

`flashinfer-bench leaderboard` is a web-based leaderboard for benchmarking CUDA kernel generation tasks. It displays models, frameworks, and their associated kernels, linking to performance leaderboards.

---

### Features

✅ Models & Frameworks overview page

✅ Model detail page with kernels and leaderboard links

✅ Leaderboards showing top users and runtimes

---

### How to run the leaderboard

1️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

```bash
npm install
```

2️⃣ Run the pipeline and start the server:

```bash
python run_pipeline.py
```

You should see:

```
 * Running on http://127.0.0.1:5000/
```


3️⃣ Open your browser:

```
http://127.0.0.1:5000/
```

✅ Navigate through Models & Frameworks, view kernels, and leaderboard pages.

---

## How to run the Inspector frontend

The Inspector is a React-based web application for viewing and analyzing kernel generation logs in JSONL format.

1️⃣ Navigate to the inspector directory:

```bash
cd inspector
```

2️⃣ Install dependencies:

```bash
npm install
```

3️⃣ Start the development server:

```bash
npm run dev
```

You should see:

```
  VITE v4.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

4️⃣ Open your browser:

```
http://localhost:3000/
```

5️⃣ Upload a `.jsonl` file containing kernel generation logs to view:
   - **Problem Description**: The original problem prompt
   - **Generated Kernel**: All generated CUDA kernels from iterations
   - **Trace**: Complete JSON log of the generation process

---

## How to run benchmark on single kernel description

**Basic usage:**

```bash
python benchmark/benchmark.py <kernel_description.py> <kernel_generator.py>
```

**With advanced arguments:**

```bash
python benchmark/benchmark.py kernel_description.py kernel_generator.py \
    --warmup 5 \
    --iter 10 \
    --report-n 16 \
    --max-diff-limit 1e-5 \
    --output benchmarking_results.json
```

## How to run benchmark on datasets

**Basic usage:**

```bash
python run_benchmark.py <kernel_generator.py> --dataset-dir <path/to/dataset>
```

**Example:**

```bash
python run_benchmark.py agents/kernelllm_generator.py --dataset-dir dataset/kernelbench/level1
```

**Available arguments:**

- `generator` - Path to kernel generator file (.py) **(required)**
- `--dataset-dir` - Directory containing datapoint files to benchmark **(required)**
- `--warmup` - Number of warmup iterations (default: 5)
- `--iter` - Number of timing iterations per round (default: 10)
- `--report-n` - Number of generation rounds (default: 16)
- `--max-diff-limit` - Maximum difference for correctness (default: 1e-5)
- `--correctness-trials` - Number of correctness trials with different inputs (default: 1)
- `--seed` - Random seed for reproducibility (default: 42)
- `--device` - CUDA device to use (e.g., 'cuda:0', 0, or 'auto')
- `--backend` - Backend to use: cuda, triton (default: cuda)
- `--loader-type` - Type of kernel loader: auto, kernelbench, triton, flashinfer (default: auto)
- `--use-ncu` - Enable NCU profiling (default: false)
- `--output` - Output aggregated results JSON file (default: benchmark_results/aggregated_results.json)
- `--benchmark-script` - Path to the benchmark.py script (default: benchmark/benchmark.py)
- `--timeout` - Timeout per datapoint in seconds (default: 1800)
- `--verbose` - Enable verbose output
- `--keep-temp` - Keep temporary directory for debugging

**Advanced example:**

```bash
python run_benchmark.py kernels/flashinfer_generator.py \
    --dataset-dir dataset/kernelbench/level1 \
    --loader-type kernelbench \
    --warmup 10 \
    --iter 20 \
    --report-n 32 \
    --backend triton \
    --device cuda:0 \
    --verbose \
    --keep-temp \
    --output benchmark_results/flashinfer_results.json
```

This will run benchmarks on all datapoint files in the specified dataset directory and save aggregated results to `benchmark_results/flashinfer_results.json`.

## ⚠️ Notes

* The site relies on precomputed `leaderboard_[id].json` and model/kernel JSONL data in `leaderboard/leaderboard/static/`. Ensure these files exist and are updated by your data processing scripts under `leaderboard/scripts/`. The traces data should be in `leaderboard/public/data`.
* Tailwind CSS must be built and linked properly in `leaderboard/leaderboard/static/css/main.css`.