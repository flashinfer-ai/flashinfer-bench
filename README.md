# flashinfer-bench leaderboard

`flashinfer-bench leaderboard` is a web-based leaderboard for benchmarking CUDA kernel generation tasks. It displays models, frameworks, and their associated kernels, linking to performance leaderboards.

---

## Features

✅ Models & Frameworks overview page

✅ Model detail page with kernels and leaderboard links

✅ Leaderboards showing top users and runtimes

✅ Inspector frontend for viewing kernel generation logs

---

## How to run the leaderboard

1️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

```bash
npm install
```

2️⃣ Start the server:

```bash
python -m leaderboard.run
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

## How to run benchmarks

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

## ⚠️ Notes

* The site relies on precomputed `leaderboard_[id].json` and model/kernel JSONL data in `leaderboard/leaderboard/static/`. Ensure these files exist and are updated by your data processing scripts under `leaderboard/scripts/`. The traces data should be in `leaderboard/public/data`.
* Tailwind CSS must be built and linked properly in `leaderboard/leaderboard/static/css/main.css`.