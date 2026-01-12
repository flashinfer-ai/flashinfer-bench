import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Literal, Optional

from flashinfer_bench import Solution, Workload, Definition
from flashinfer_bench.compile import BuilderRegistry


SanitizerType = Literal["memcheck", "racecheck", "initcheck", "synccheck"]


def _run_sanitizer(
    definition: Definition,
    solution: Solution,
    workload: Workload,
    sanitizer_types: Optional[List[SanitizerType]] = None,
) -> str:
    if sanitizer_types is None:
        sanitizer_types = ["memcheck", "racecheck", "initcheck", "synccheck"]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            script_path = temp_path / "run_kernel.py"
            script_content = _generate_test_script(definition, solution, workload, temp_path)
            script_path.write_text(script_content)
            
            for sanitizer_type in sanitizer_types:
                output += f"\n{'='*60}\n"
                output += f"Running {sanitizer_type.upper()}\n"
                output += f"{'='*60}\n\n"

                sanitizer_cmd = [
                    "compute-sanitizer",
                    "--tool", sanitizer_type,
                    sys.executable,
                    str(script_path),
                ]

                try:
                    result = subprocess.run(
                        sanitizer_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 min
                        cwd=str(temp_path),
                    )

                    output += f"STDOUT:\n{result.stdout}\n\n"

                    if result.stderr:
                        output += f"STDERR:\n{result.stderr}\n\n"

                    output += f"Return code: {result.returncode}\n"

                    if result.returncode != 0:
                        output += f"\nWARNING: {sanitizer_type} detected issues!\n"
                    else:
                        output += f"\n{sanitizer_type} passed successfully.\n"

                except subprocess.TimeoutExpired:
                    output += f"Error: {sanitizer_type} timed out (5 minutes)\n"
                except FileNotFoundError:
                    output += f"Error: compute-sanitizer command not found. Please ensure CUDA toolkit is installed.\n"
                    break
                except Exception as e:
                    output += f"Error running {sanitizer_type}: {str(e)}\n"            
            
        except Exception as e:
            return f"Error building solution: {str(e)}"

    output += f"\n{'='*60}\n"
    output += "Sanitizer checks complete\n"
    output += f"{'='*60}\n"

    return output


def _generate_test_script(
    definition: Definition,
    solution: Solution,
    workload: Workload,
    temp_path: Path,
) -> str:
    (temp_path / "definition.json").write_text(definition.model_dump_json(indent=2))
    (temp_path / "solution.json").write_text(solution.model_dump_json(indent=2))
    (temp_path / "workload.json").write_text(workload.model_dump_json(indent=2))
    
    script = '''
from pathlib import Path
import torch
from flashinfer_bench import Definition, Solution, Workload
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.utils import dtype_str_to_torch_dtype

def main():
    base_path = Path(__file__).parent
    
    with open(base_path / "definition.json") as f:
        definition = Definition.model_validate_json(f.read())
    with open(base_path / "solution.json") as f:
        solution = Solution.model_validate_json(f.read())
    with open(base_path / "workload.json") as f:
        workload = Workload.model_validate_json(f.read())
    
    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)
    
    device = "cuda"
    
    safe_tensors = None
    if any(inp.type == "safetensors" for inp in workload.inputs.values()):
        safe_tensors = load_safetensors(definition, workload, trace_set_root=base_path)    
    inputs = gen_inputs(definition, workload, device, safe_tensors)
    
    outputs = []
    output_shapes = definition.get_output_shapes(workload.axes)
    for (output_name, output_spec), shape in zip(definition.outputs.items(), output_shapes):
        dtype = dtype_str_to_torch_dtype(output_spec.dtype)
        if shape is None:
            tensor = torch.empty((), dtype=dtype, device=device)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        outputs.append(tensor)
    
    for _ in range(3):
        if solution.spec.destination_passing_style:
            runnable(*inputs, *outputs)
        else:
            result = runnable(*inputs)
    
    torch.cuda.synchronize()    
    runnable.cleanup()

if __name__ == "__main__":
    main()
'''
    return script

def flashinfer_bench_run_sanitizer(
    definition: Definition, solution: Solution, workload: Workload  
) -> str:
    """Runs all sanitizer checks: memcheck, racecheck, initcheck, synccheck"""
    return _run_sanitizer(definition, solution, workload)

def flashinfer_bench_run_memcheck(
    definition: Definition, solution: Solution, workload: Workload  
) -> str:
    return _run_sanitizer(definition, solution, workload, ["memcheck"])


def flashinfer_bench_run_racecheck(
    definition: Definition, solution: Solution, workload: Workload
) -> str:
    return _run_sanitizer(definition, solution, workload, ["racecheck"])


def flashinfer_bench_run_initcheck(
    definition: Definition, solution: Solution, workload: Workload
) -> str:
    return _run_sanitizer(definition, solution, workload, ["initcheck"])


def flashinfer_bench_run_synccheck(
    definition: Definition, solution: Solution, workload: Workload
) -> str:
    return _run_sanitizer(definition, solution, workload, ["synccheck"])
