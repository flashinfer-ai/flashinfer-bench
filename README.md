# FlashInfer Bench

**FlashInfer Bench** is a lightweight, extensible benchmarking suite for evaluating low-level kernel implementations of model inference workloads. It is centered around the `Trace` artifact — a detailed record of a workload execution. It enables systematic comparison of kernel implementations with correctness and performance metrics.

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

## Benchmarking Kernels

You can run local benchmarks using the `Benchmark` runner, which scans your dataset for all available definitions and solutions, executes them, and appends resulting traces to the `TraceSet`.

It also supports single-solution execution via `.run_solution(...)`.

```python
from flashinfer_bench import Benchmark, TraceSet

traces = TraceSet.from_path("./dataset")
benchmark = Benchmark(traces)

benchmark.run()
```

## Schema

Each of the core entities is modeled as a dataclass:

* **Definition**: Workload specification with axes, inputs, outputs, and a reference implementation.
* **Solution**: A concrete implementation with source files and a launch entry point.
* **Trace**: A benchmark result of a solution on a specific workload input.

See [`schema/`](./schema/) for full documentation.

