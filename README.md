# FlashInfer Bench

**FlashInfer Bench** is a lightweight, extensible benchmarking suite for evaluating low-level kernel implementations of model inference workloads. It is centered around the `Trace` artifact — a detailed record of a workload execution. It enables systematic comparison of kernel implementations with correctness and performance metrics.

## Installation

Install FlashInfer Bench with pip:

```bash
pip install -v -e .
``` 

## Dataset Layout

Each dataset is organized as follows:

```
dataset/
├── definitions/         # One JSON file per workload definition
├── solutions/           # One JSON file per solution implementation
└── traces/              # Benchmark results
```

* Each **Definition** describes a computation task and reference logic.
* Each **Solution** specifies a kernel or agent implementation for a definition.
* Each **Trace** records a benchmark result: input config, performance, correctness, environment, etc.

You can load the full dataset using:

```python
from flashinfer_bench import TraceSet
trace_set = TraceSet.from_path("/dataset")
```

## Command Line Interface (CLI)

FlashInfer Bench provides a CLI for running benchmarks and analyzing results.

### Usage

#### Options
- `--local <PATH>`: Specifies one or more local paths to load traces from. Can be used multiple times.
- `--hub`: Load the latest traces from the FlashInfer Hub (not yet implemented).
- `--warmup-runs <N>`: Number of warmup runs for benchmarking (default: 10).
- `--iterations <N>`: Number of benchmark iterations (default: 50).
- `--device <DEVICE>`: Device to run benchmarks on (default: cuda:0).
- `--log-level <LEVEL>`: Logging level (default: INFO).
- `--save-results` / `--no-save-results`: Whether to save results after running (default: save).

#### Example

```bash
# Run benchmarks on a dataset
flashinfer-bench run --local ./dataset

# Print a summary of traces
flashinfer-bench report summary --local ./dataset

# Find the best solution for each definition
flashinfer-bench report best --local ./dataset
```

## Benchmarking Kernels

You can run local benchmarks using the `Benchmark` runner, which scans your dataset for all available definitions and solutions, executes them, and appends resulting traces to the `TraceSet`.

It also supports single-solution execution via `.run_solution(...)`.

```python
from flashinfer_bench import Benchmark, TraceSet

traces = TraceSet.from_path("./dataset")
benchmark = Benchmark(traces)

benchmark.run()

# Accessing results
print(traces.summary())

# Customizing the benchmark run
from flashinfer_bench import BenchmarkConfig
config = BenchmarkConfig(warmup_runs=5, iterations=20, device="cuda:1")
benchmark.run(config)
```

## Schema

Each of the core entities is modeled as a dataclass:

* **Definition**: Workload specification with axes, inputs, outputs, and a reference implementation.
* **Solution**: A concrete implementation with source files and a launch entry point.
* **Trace**: A benchmark result of a solution on a specific workload input.

See [`schema/`](./schema/) for full documentation.

