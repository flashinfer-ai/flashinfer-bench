# examples/benchmark_runner.py

from flashinfer_bench import Benchmark

def main():
    benchmark = Benchmark.from_path("./dataset")
    db = benchmark.run()

    print(f"\nCompleted benchmarking. Found {len(db.traces)} traces.")
    
    print("\nTraces:")
    for i, trace in enumerate(db.traces):
        print(f"\nTrace {i+1}:")
        print(trace)

if __name__ == "__main__":
    from flashinfer_bench import Trace
    main()
