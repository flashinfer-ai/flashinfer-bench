"""
This file contains the prompts for baseline agent generation.
"""

from flashinfer_bench import Definition, EvaluationStatus, Trace


def _format_definition(definition: Definition) -> str:
    axes_str = "\nAxes:\n"
    for name, axis in definition.axes.items():
        if hasattr(axis, "value"):
            axes_str += f"  {name}: constant = {axis.value}"
        else:
            axes_str += f"  {name}: variable"
        if axis.description:
            axes_str += f" ({axis.description})"
        axes_str += "\n"

    # Format inputs
    inputs_str = "\nInputs:\n"
    for name, spec in definition.inputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        inputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            inputs_str += f" - {spec.description}"
        inputs_str += "\n"

    outputs_str = "\nOutputs:\n"
    for name, spec in definition.outputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        outputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            outputs_str += f" - {spec.description}"
        outputs_str += "\n"

    constraints_str = ""
    if definition.constraints:
        constraints_str = "\nConstraints:\n"
        for constraint in definition.constraints:
            constraints_str += f"  - {constraint}\n"

    return f"""Name: {definition.name}
Type: {definition.op_type}
{axes_str}{inputs_str}{outputs_str}{constraints_str}

Reference Implementation:
{definition.reference}"""


def _format_trace_logs(trace: Trace) -> str:
    if trace.is_workload_trace() or not trace.evaluation:
        return "No evaluation logs available (workload-only trace)"

    eval_info = f"Status: {trace.evaluation.status.value}\n"
    eval_info += f"Timestamp: {trace.evaluation.timestamp}\n"

    if trace.evaluation.log:
        eval_info += f"\nExecution Log:\n{trace.evaluation.log}\n"

    if trace.evaluation.correctness:
        eval_info += f"Max relative error: {trace.evaluation.correctness.max_relative_error}\n"
        eval_info += f"Max absolute error: {trace.evaluation.correctness.max_absolute_error}\n"

    if trace.evaluation.performance:
        eval_info += f"Latency: {trace.evaluation.performance.latency_ms}ms\n"
        eval_info += f"Reference latency: {trace.evaluation.performance.reference_latency_ms}ms\n"
        eval_info += f"Speedup factor: {trace.evaluation.performance.speedup_factor}x\n"

    return eval_info


TRITON_PROMPT = """Generate a Triton kernel optimized for {target_gpu} GPU for

{definition}

Triton Version: 3.3.1

Requirements:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your Triton implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- Use math.log(2), math.pi, math.e instead of hex literals
- All code must be valid Python that passes ast.parse()

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the code, no explanations or markdown formatting

Generate complete, runnable code only - no framework will add device handling wrapper code.

Generate the implementation:"""

TRITON_OPTIMIZATION_PROMPT = """You are optimizing a Triton kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

Optimization Strategy:
1. ENSURE CORRECTNESS: If there are compile errors, runtime errors, or incorrect outputs, focus entirely on fixing these issues
   - Analyze compilation errors and fix syntax/API usage
   - Fix runtime errors like shape mismatches, memory access violations
   - Ensure numerical correctness matches the reference implementation

2. OPTIMIZE PERFORMANCE: if the current kernel is functionally correct, focus on performance optimizations
   - Optimize memory access patterns for {target_gpu}
   - Tune block sizes and grid dimensions
   - Use appropriate Triton language features for vectorization
   - Minimize global memory transactions

Requirements for the optimized implementation:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Fix all identified issues from the feedback
- Maintain or improve computational accuracy
- Preserve the same function signature and device handling as specified

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- Use math.log(2), math.pi, math.e instead of hex literals
- All code must be valid Python that passes ast.parse()

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the improved code, no explanations or markdown formatting

Generate the corrected and optimized implementation:"""

PYTHON_PROMPT = """You are a code generator. Generate a Python implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

Requirements:
- Write clean, efficient Python code optimized for {target_gpu} architecture
- Use PyTorch operations when appropriate, optimized for {target_gpu}
- Include necessary imports
- Implement the exact functionality described in the specification
- Expose a "run" entry point function that can be called to execute the implementation
- Return only the code, no explanations or markdown formatting

Generate the implementation:"""

# CUDA prompt
CUDA_PROMPT = """You are a code generator. Generate a CUDA kernel implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

Requirements:
- Write clean, efficient CUDA C++ code optimized for {target_gpu} architecture
- Use proper CUDA syntax and memory management optimized for {target_gpu}
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your CUDA implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)
- For fixed axis values, optimize specifically for those constants rather than general cases
- You may use 3rd party libraries (cuBLAS, cuDNN, CUTLASS) when beneficial, but custom implementations often perform better for specialized kernels with known axis constraints

IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
- Proper error checking and memory management
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Memory allocation and data transfer management
- Device management and error handling
- Entry point function named "run" that can be called to execute the implementation
- Handle both args and kwargs properly
- Move CPU data to GPU, execute kernels, and return results to CPU
</cpp_file>

Code Generation Guidelines:
- Use modern CUDA features appropriate for {target_gpu}
- Optimize memory coalescing and reduce bank conflicts
- Utilize shared memory effectively for data reuse
- Consider occupancy and register usage
- Implement proper error checking with cudaGetLastError()
- Use appropriate grid and block dimensions for the problem size
- Leverage constant memory for frequently accessed read-only data
- Ensure proper CUDA stream synchronization and error handling

Generate the implementation:"""

CUDA_OPTIMIZATION_PROMPT = """You are optimizing a CUDA kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

Optimization Strategy:
1. ENSURE CORRECTNESS: If there are compile errors, runtime errors, or incorrect outputs, focus entirely on fixing these issues
   - Analyze compilation errors and fix syntax/API usage
   - Fix runtime errors like shape mismatches, memory access violations, kernel launch failures
   - Ensure numerical correctness matches the reference implementation
   - Verify proper CUDA memory management and synchronization

2. OPTIMIZE PERFORMANCE: if the current kernel is functionally correct, focus on performance optimizations
   - Optimize memory access patterns and coalescing for {target_gpu}
   - Tune block sizes and grid dimensions for maximum occupancy
   - Utilize shared memory effectively to reduce global memory transactions
   - Optimize register usage and minimize divergent branches
   - Consider using specialized libraries (cuBLAS, cuDNN, CUTLASS) where beneficial
   - Leverage constant axis values for compile-time optimizations

Requirements for the optimized implementation:
- Write clean, efficient CUDA C++ code optimized for {target_gpu} architecture
- Use proper CUDA syntax and modern features appropriate for {target_gpu}
- Fix all identified issues from the feedback
- Maintain or improve computational accuracy
- Preserve the same function signatures and device handling as specified
- For fixed axis values, optimize specifically for those constants rather than general cases

IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
- Proper error checking and memory management
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Memory allocation and data transfer management
- Device management and error handling
- Entry point function named "run" that can be called to execute the implementation
- Handle both args and kwargs properly
- Move CPU data to GPU, execute kernels, and return results to CPU
</cpp_file>

Code Generation Guidelines:
- Use modern CUDA features appropriate for {target_gpu}
- Optimize memory coalescing and reduce bank conflicts
- Utilize shared memory effectively for data reuse
- Consider occupancy and register usage
- Implement proper error checking with cudaGetLastError()
- Use appropriate grid and block dimensions for the problem size
- Leverage constant memory for frequently accessed read-only data
- Ensure proper CUDA stream synchronization and error handling

Generate the corrected and optimized implementation:"""

TORCH_BINDINGS_PROMPT = """
Use TORCH for your generated kernel host function and bindings

Requirements:
- Include all necessary headers (torch/extension.h, kernel.h, etc.)
- Implement the "run" function that:
  * Takes torch::Tensor arguments
  * Validates tensor properties (device, dtype, shape)
  * Extracts raw pointers using .data_ptr<T>()
  * Calls the CUDA kernel with appropriate launch configuration
  * Returns results as torch::Tensor
- Use PYBIND11_MODULE to bind the "run" function:
  * PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  *   m.def("run", &run, "Kernel execution function");
  * }}
- Handle both positional args and kwargs properly
- Include proper error messages for invalid inputs

- Use torch::Tensor for all tensor arguments
- Use .device().is_cuda() to check if tensors are on GPU
- Use .dtype() to validate tensor data types
- Use .sizes() or .size(dim) to get tensor dimensions
- Use .data_ptr<float>() or .data_ptr<T>() to get raw pointers
- Call cudaDeviceSynchronize() or cudaGetLastError() for error checking
- Return torch::Tensor from the run function
- Handle exceptions gracefully with proper error messages"""


def get_prompt(language: str, definition: Definition, target_gpu: str = "H100") -> str:
    prompts = {"triton": TRITON_PROMPT, "python": PYTHON_PROMPT, "cuda": CUDA_PROMPT}

    if language not in prompts:
        raise ValueError(f"Unsupported language: {language}")

    definition_str = _format_definition(definition)

    return prompts[language].format(definition=definition_str, target_gpu=target_gpu)


def get_optimization_prompt(
    language: str, definition, trace: Trace, current_code: str, target_gpu: str = "H100"
) -> str:
    optimization_prompts = {"triton": TRITON_OPTIMIZATION_PROMPT, "cuda": CUDA_OPTIMIZATION_PROMPT}

    if language not in optimization_prompts:
        raise ValueError(f"Unsupported language for optimization: {language}")

    definition_str = _format_definition(definition)
    trace_logs = _format_trace_logs(trace)

    return optimization_prompts[language].format(
        definition=definition_str,
        trace_logs=trace_logs,
        current_code=current_code,
        target_gpu=target_gpu,
    )

