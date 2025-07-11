# examples/benchmark_runner.py

from flashinfer_bench import Benchmark

def main():
    benchmark = Benchmark.from_path("./dataset")
    db = benchmark.run()

    print(f"\nCompleted benchmarking. Found {len(db.traces)} traces.")
    
    # Save all traces to disk
    output_path = "./dataset/traces_benchmark_test.jsonl"
    Trace.save_jsonl(output_path, db.traces)
    print(f"Traces saved to {output_path}")

if __name__ == "__main__":
    from flashinfer_bench.specs.trace import Trace
    main()
