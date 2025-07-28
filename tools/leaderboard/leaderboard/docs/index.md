# Welcome to the FlashInfer Bench Documentation

**FlashInfer Bench** is a lightweight and extensible benchmarking suite designed to evaluate low-level kernel implementations for model inference workloads. By capturing detailed execution records — called `Trace` — it provides a principled framework for comparing implementations with respect to both performance and correctness.

FlashInfer Bench is ideal for researchers, and ML practitioners who need reproducible benchmarks and systematic analysis of kernel behavior across diverse configurations and environments.


## About This Project

This documentation provides guidance on how to use FlashInfer Bench, how datasets are organized, how to run benchmarks and generate reports, and how to contribute new kernels or workloads.

The official repository is hosted at [mlc-ai/flashinfer-bench](https://github.com/mlc-ai/flashinfer-bench).


## Getting Started

To begin using FlashInfer Bench, install it via pip:

```bash
pip install -v -e .
```

After installation, you can interact with the suite either through Python or the command-line interface (CLI).

## Dataset Structure

FlashInfer Bench operates on structured datasets containing definitions, solutions, and traces:

```bash
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


### Basic Usage

Run benchmarks locally:

```bash
flashinfer-bench run --local ./dataset
```

Summarize trace results:

```bash
flashinfer-bench report summary --local ./dataset
```

Identify the best-performing solutions:

```bash
flashinfer-bench report best --local ./dataset
```

### Available Options

* `--local <PATH>`: Load trace data from one or more local paths.
* `--hub`: Load traces from the FlashInfer Hub (feature in progress).
* `--warmup-runs <N>`: Number of warmup runs before measurement (default: 10).
* `--iterations <N>`: Number of benchmark iterations (default: 50).
* `--device <DEVICE>`: Target device for evaluation (e.g., `cuda:0`).
* `--log-level <LEVEL>`: Set logging verbosity (default: `INFO`).
* `--save-results` / `--no-save-results`: Enable or disable writing results to disk.


## Benchmarking Kernels via Python

For advanced use or custom workflows, you can invoke benchmarking directly from Python:

```python
from flashinfer_bench import Benchmark, TraceSet

traces = TraceSet.from_path("./dataset")
benchmark = Benchmark(traces)
benchmark.run()

# View summary statistics
print(traces.summary())
```

You can also configure the benchmark process:

```python
from flashinfer_bench import BenchmarkConfig

config = BenchmarkConfig(warmup_runs=5, iterations=20, device="cuda:1")
benchmark.run(config)
```

## Schema Overview

FlashInfer Bench is built around three key data abstractions:

* **Definition**: Specifies the input/output semantics and reference logic for a workload.
* **Solution**: Points to an implementation, such as a Triton or CUDA kernel, with metadata and source files.
* **Trace**: Represents the result of executing a solution on a definition under specific inputs and settings.

Each of these is implemented as a structured dataclass. See [`schema`](https://github.com/mlc-ai/flashinfer-bench/tree/main/schema) for detailed specifications and extension guidelines.

---

## Why FlashInfer Bench?

Traditional kernel benchmarks are often ad hoc and hard to reproduce or compare. FlashInfer Bench addresses this by:

* Standardizing inputs and outputs across workloads
* Supporting both local and collaborative trace datasets
* Enabling trace-driven analysis across implementations and workloads

By capturing the full context of execution, FlashInfer Bench empowers you to draw meaningful insights from low-level benchmarks — whether you're tuning Triton/CUDA kernels, validating correctness, or comparing vendor backends.