"""
Example script to generate optimized solutions with KernelGenerator module
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from kernel_generator import KernelGenerator

from flashinfer_bench import TraceSet
from flashinfer_bench.data import save_json_file

load_dotenv()


def main():
    """
    Generate optimized solutions for all definitions in the trace_set.
    """
    # TODO: select model, language, target gpu, definition
    model_name = "gpt-5-2025-08-07"  # Choose author-model
    language = "triton"  # Target solution language
    target_gpu = "B200"  # Choose solution target GPU
    target_definition_name = ""  # Leave empty to generate solutions for all definitions

    # TODO: adjust local path to trace_set
    trace_set_path = "/path/to/flashinfer-trace"

    print(f"Loading TraceSet from: {trace_set_path}")
    trace_set = TraceSet.from_path(trace_set_path)

    all_definitions = list(trace_set.definitions.keys())

    if not all_definitions:
        print(f"Error: No definitions found in trace_set at '{trace_set_path}'.")
        print("Please ensure `trace_set_path` points to a valid flashinfer-trace directory.")
        return

    if target_definition_name:
        if target_definition_name in all_definitions:
            all_definitions = [target_definition_name]
            print(f"Generating solution {target_definition_name}")
        else:
            print(f"Definition '{target_definition_name}' not found in trace_set")
            return

    print(f"Found {len(all_definitions)} definitions to generate solutions")

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
        use_ffi=False,
    )

    total_definitions = len(all_definitions)
    successful_generations = 0
    failed_generations = 0

    print(f"\n{'='*60}")
    print(f"Generating solutions for {total_definitions} definitions...")
    print(f"{'='*60}")

    for idx, definition_name in enumerate(all_definitions, 1):
        definition = trace_set.definitions[definition_name]

        print(f"\n[{idx}/{total_definitions}] Processing definition: {definition_name}")
        print(f"Definition type: {definition.op_type}")

        workloads = trace_set.workloads.get(definition_name, [])
        if not workloads:
            print(f"No workloads found for definition '{definition_name}' - SKIPPING")
            failed_generations += 1
            continue

        print(f"Found {len(workloads)} workloads for this definition")

        solution = None
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            try:
                print(f"\nAttempt {attempt}/{max_attempts} for {definition_name}")

                solution = generator.generate(
                    trace_set=trace_set,
                    definition=definition,
                    gen_rounds=10,  # For our baseline, we used 10 rounds
                    # TODO: uncomment bellow to use beam search
                    # beam=True,
                    # beam_width=3,
                )

                print(f"Successfully generated solution for {definition_name}")
                break

            except Exception as e:
                print(f"Attempt {attempt} failed for {definition_name}: {e}")
                if attempt < max_attempts:
                    print(f"Retrying... ({attempt + 1}/{max_attempts})")
                else:
                    print(f"All attempts failed for {definition_name} - SKIPPING")
                    failed_generations += 1
                    break

        if solution:
            try:
                # Create directory structure: solutions/definition-type/definition-name/
                solutions_dir = (
                    Path(trace_set_path) / "solutions" / definition.op_type / definition_name
                )
                solutions_dir.mkdir(parents=True, exist_ok=True)

                # Create filename using solution name
                solution_filename = f"{solution.name}.json"
                solution_path = solutions_dir / solution_filename

                save_json_file(solution, solution_path)

                print(f"Solution saved to: {solution_path}")
                successful_generations += 1

            except Exception as e:
                print(f"Failed to save solution for {definition_name}: {e}")
                failed_generations += 1

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total definitions processed: {total_definitions}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    success_rate = (successful_generations / total_definitions * 100) if total_definitions else 0.0
    print(f"Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()
