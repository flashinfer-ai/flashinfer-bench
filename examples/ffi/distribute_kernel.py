import shutil
from pathlib import Path

from flashinfer_bench import TraceSet
from flashinfer_bench.compile.builders.tvm_ffi_builder import TVMFFIBuilder


def main():
    traceset = TraceSet.from_path("Example-FlashInfer-Trace")

    definition_name = "gemm_n4096_k4096"
    definition = traceset.definitions[definition_name]

    solutions = list(traceset.solutions[definition_name])
    solution = solutions[0]
    print(f"Building solution: {solution.name}")

    builder = TVMFFIBuilder()
    runnable = builder.build(definition, solution)

    so_path = runnable.meta["binary"]
    entry_symbol = runnable.meta["symbol"]

    dist_dir = Path("distributed")
    dist_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so_path, dist_dir / "kernel.so")

    with open(dist_dir / "kernel_metadata.txt", "w") as f:
        f.write(f"Entry Symbol: {entry_symbol}\n")
        f.write(f"Definition: {definition.name}\n")
        f.write(f"Solution: {solution.name}\n")

    print(f"Built kernel: {dist_dir / 'kernel.so'}")
    print(f"Entry symbol: {entry_symbol}")


if __name__ == "__main__":
    main()
