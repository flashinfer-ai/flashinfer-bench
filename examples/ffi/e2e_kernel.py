"""
End-to-end example script, runs benchmark on agent generated gemm kernel, and applies it to a test input.
"""
import torch
from flashinfer_bench import (
    Benchmark,
    BenchmarkConfig,
    TraceSet,
    apply,
    enable_apply,
    disable_apply,
)

def main():
    traceset_path = "Example-FlashInfer-Trace"
    
    print(f"Loading Example-FlashInfer-Trace")
    traceset = TraceSet.from_path(traceset_path)
    
    # Run benchmark
    print("\nRunning Benchmark on agent generated gemm kernel")
    benchmark = Benchmark(traceset, BenchmarkConfig())
    result_traceset = benchmark.run_all()
    
    print("\nBenchmark Complete")
    for def_name, traces in result_traceset.traces.items():
        print(f"\n{def_name}:")
        for trace in traces:
            print(f"  Solution: {trace.solution.name}")
            print(f"  Status: {trace.evaluation.status.value}")
            if trace.evaluation.performance:
                print(f"  Speedup: {trace.evaluation.performance.speedup_factor:.2f}x")
    
    # Apply kernel
    print("\nApplying generated kernel on test input")
    
    with enable_apply(traceset_path):
        # Random test inputs for gemm_n4096_k4096
        # A: [M, K] = [1024, 4096]
        # B: [N, K] = [4096, 4096]
        # C = A @ B.T -> [1024, 4096]
        
        M, N, K = 1024, 4096, 4096        
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(N, K, dtype=torch.float16, device="cuda")
        
        def reference_gemm(A, B):
            return torch.matmul(A, B.T)        
        ref_sol = reference_gemm(A, B)
        
        test_sol = apply(
            "gemm_n4096_k4096",
            runtime_kwargs={"A": A, "B": B},
            fallback=reference_gemm
        )
        
        if test_sol is not None:
            diff = torch.abs(ref_sol - test_sol).max().item()
            print(f"Max difference vs reference: {diff}")
            if diff < 1e-2:
                print("Vibe coded kernel is correct")
            else:
                print("Vibe coded kernel differs from torch matmul")
        else:
            print("No result returned (using fallback)")    

if __name__ == "__main__":
    main()

