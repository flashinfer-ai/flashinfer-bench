# FlashInfer Bench Leaderboard

A web leaderboard for visualizing performance from FlashInfer Bench. This project reads static traces from `/flashinfer-bench/dataset` and presents per-kernel rankings grouped by hardware device.

## Features

- Ranks kernel implementations by runtime (latency in ms)
- Groups results by hardware device
- Uses static `.jsonl` files — no database required

## Running the Leaderboard

### 1. Navigate to the Leaderboard Directory

```bash
cd tools/leaderboard
```

### 2. Set up a virtual environment (optionally) and install dependencies

Optional:
```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the App

```bash
python -m leaderboard.run
```

Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)


## Dataset Format

Benchmark trace files are read from a static directory (e.g., `/flashinfer-bench/dataset/`) and must follow the `TraceSet` schema, with:

* `definitions/*.json`
* `solutions/*.json`
* `traces/*.jsonl`

For complete schema documentation, see `/flashinfer-bench/schema/`

This schema defines how `definitions`, `solutions`, and `traces` are structured and used to build the leaderboard.


## Extend the Leaderboard

To add new definitions or solutions, simply place the JSON files in the appropriate directories (under `/flashinfer-bench/dataset/`), and run the benchmark. The leaderboard will automatically pick them up.