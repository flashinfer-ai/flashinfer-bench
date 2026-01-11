import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from flashinfer_bench import TraceSet
from flashinfer_bench.data import save_json_file

# Get the path to examples/kernel_generator
script_dir = Path(__file__).parent  # examples/ffi
examples_dir = script_dir.parent  # examples
kernel_gen_dir = examples_dir / "kernel_generator"

sys.path.insert(0, str(kernel_gen_dir))
from kernel_generator import KernelGenerator

load_dotenv(kernel_gen_dir / ".env")


def main():
    """
    Generate optimized CUDA solutions for FFI bindings.
    """
    # TODO: select model, target gpu, definition
    model_name = "gpt-5-2025-08-07"  # Choose model
    language = "cuda"  # Target CUDA for FFI bindings
    target_gpu = "B200"  # Target GPU

    print(f"Loading Example TraceSet")
    traceset_path = script_dir / "Example-FlashInfer-Trace"
    traceset = TraceSet.from_path(traceset_path)

    definition_name = "gemm_n4096_k4096"
    definition = traceset.definitions[definition_name]

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key:
        print(
            "Please set LLM_API_KEY environment variable or modify this script to pass api_key explicitly"
        )
        return

    generator = KernelGenerator(
        model_name=model_name,
        language=language,
        target_gpu=target_gpu,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort="high",
        use_ffi=True,
    )

    print(f"\n{'='*60}")
    print(f"Generating CUDA solution for: {definition_name}")
    print(f"Definition type: {definition.op_type}")
    print(f"Target GPU: {target_gpu}")
    print(f"{'='*60}")

    # Get workloads for this definition
    workloads = traceset.workloads.get(definition_name, [])
    if not workloads:
        print(f"No workloads found for definition '{definition_name}'")
        return

    print(f"Found {len(workloads)} workloads for this definition")

    # Generate solution with beam search
    solution = None
    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"\nAttempt {attempt}/{max_attempts}")

            solution = generator.generate(
                traceset=traceset,
                definition=definition,
                gen_rounds=10,  # search depth
                beam=True,
                beam_width=3,
            )

            print(f"Successfully generated solution for {definition_name}")
            break

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                print(f"Retrying... ({attempt + 1}/{max_attempts})")
            else:
                print(f"All attempts failed - aborting")
                return

    # Save the solution
    if solution:
        try:
            solutions_dir = Path(traceset_path) / "solutions"
            solutions_dir.mkdir(parents=True, exist_ok=True)

            solution_filename = f"{solution.name}.json"
            solution_path = solutions_dir / solution_filename

            save_json_file(solution, solution_path)

            print(f"\n{'='*60}")
            print(f"SUCCESS!")
            print(f"{'='*60}")
            print(f"Solution saved to: {solution_path}")

        except Exception as e:
            print(f"Failed to save solution: {e}")
            return


if __name__ == "__main__":
    main()
