# flashinfer-bench leaderboard

`flashinfer-bench leaderboard` is a web-based leaderboard for benchmarking CUDA kernel generation tasks. It displays models, frameworks, and their associated kernels, linking to performance leaderboards.

---

## Features

✅ Models & Frameworks overview page

✅ Model detail page with kernels and leaderboard links

✅ Leaderboards showing top users and runtimes

---

## How to run

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

## ⚠️ Notes

* The site relies on precomputed `leaderboard_[id].json` and model/kernel JSONL data in `leaderboard/leaderboard/static/`. Ensure these files exist and are updated by your data processing scripts under `leaderboard/scripts/`. The traces data should be in `leaderboard/public/data`.
* Tailwind CSS must be built and linked properly in `leaderboard/leaderboard/static/css/main.css`.