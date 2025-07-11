from flashinfer_bench import Database

db = Database.from_uri("dataset")

for d in db.definitions:
    print(f"\nWorkload: '{d.name}'  —  Type: {d.type}")
    print(f"  Description: {d.description or '(no description)'}")
    
    best = db.get_best_op(d.name, max_abs_diff=1e-4, max_relative_diff=1e-4)
    
    if best:
        eval = best.evaluation
        perf = eval.performance
        corr = eval.correctness
        env = eval.environment

        print(f"  Best Solution: {best.solution}")
        print(f"  - Speedup      : {perf.speedup_factor:.2f}×")
        print(f"  - Latency      : {perf.latency_ms:.4f} ms (ref: {perf.reference_latency_ms:.4f} ms)")
        print(f"  - Errors       : abs={corr.max_absolute_error:.2e}, rel={corr.max_relative_error:.2e}")
        print(f"  - Status       : {eval.status}")
        print(f"  - Device       : {env.device}")
        print(f"  - Libraries    : {', '.join(f'{k}={v}' for k, v in env.libs.items())}")
        print(f"  - Timestamp    : {eval.timestamp}")
        print(f"  - Log file     : {eval.log_file}")
    else:
        print("  No valid trace found under the current error thresholds.")
