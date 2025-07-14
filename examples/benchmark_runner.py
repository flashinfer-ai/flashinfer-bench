# examples/benchmark_runner.py

from flashinfer_bench import Benchmark, TraceSet


def main():
    traces = TraceSet.from_path("./dataset")
    benchmark = Benchmark(traces)

    initial_trace_count = sum(len(trace_list) for trace_list in benchmark.traces.values())
    print(
        f"Initial state: {len(benchmark.traces)} definition(s) with {initial_trace_count} existing traces"
    )

    benchmark.run()

    final_trace_count = sum(len(trace_list) for trace_list in benchmark.traces.values())
    new_traces = final_trace_count - initial_trace_count

    print(f"\nCompleted benchmarking!")
    print(
        f"Total: {len(benchmark.traces)} definition(s) with {final_trace_count} traces ({new_traces} new)"
    )

    print("\nDetailed Results:")
    for def_name, trace_list in benchmark.traces.items():
        print(f"\nDefinition: {def_name} ({len(trace_list)} traces)")
        for i, trace in enumerate(trace_list):
            status = trace.evaluation.get("status", "UNKNOWN")
            workload_info = trace.workload.get("axes", {})
            if status == "PASSED":
                latency = trace.evaluation.get("performance", {}).get("latency_ms", "N/A")
                speedup = trace.evaluation.get("performance", {}).get("speedup_factor", "N/A")
                print(f"  Trace {i+1}: {trace.solution} - {status} - {workload_info}")
                print(f"           Latency: {latency}ms, Speedup: {speedup}x")
            else:
                print(f"  Trace {i+1}: {trace.solution} - {status} - {workload_info}")

    summary = traces.summary()
    print(f"\nSummary: {summary['passed']}/{summary['total']} passed")
    if summary["avg_latency_ms"]:
        print(f"Average latency: {summary['avg_latency_ms']:.2f}ms")


def demo_single_solution():
    """Demo running a single solution"""
    print("\n" + "=" * 50)
    print("DEMO: Running single solution")
    print("=" * 50)

    traces = TraceSet.from_path("./dataset")
    benchmark = Benchmark(traces)

    # Run specific solution
    if traces.solutions:
        first_def = next(iter(traces.solutions.keys()))
        first_solution = traces.solutions[first_def][0]

        initial_count = sum(len(trace_list) for trace_list in benchmark.traces.values())
        print(f"\nRunning single solution: {first_solution.name}")

        benchmark.run_solution(first_solution.name)

        final_count = sum(len(trace_list) for trace_list in benchmark.traces.values())
        print(f"Added {final_count - initial_count} new traces")


if __name__ == "__main__":
    main()
    demo_single_solution()
