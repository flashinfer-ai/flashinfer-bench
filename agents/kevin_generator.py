import openai
import torch
import torch.nn as nn
import time
import re
from typing import Optional

class KevinKernelGenerator:
    def __init__(self, port: int = 30000, host: str = "127.0.0.1"):
        self.client = openai.Client(
            base_url=f"http://{host}:{port}/v1", 
            api_key="None"
        )
        self.system_prompt = """You are given the following architecture:
import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"
    Simple model that performs Layer Normalization.
    \"\"\"
    def __init__(self, normalized_shape: tuple):
        \"\"\"
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        \"\"\"
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        \"\"\"
        return self.ln(x)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100 (e.g. shared memory, kernel fusion, warp primitives, vectorization,...). Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew. You're not allowed to use torch.nn (except for Parameter, containers, and init). The input and output have to be on CUDA device. Your answer must be the complete new architecture (no testing code, no other code): it will be evaluated and you will be given feedback on its correctness and speedup so you can keep iterating, trying to maximize the speedup. After your answer, summarize your changes in a few sentences.Here's an example:

import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
\"\"\"

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)"""

    def extract_code_from_response(self, content: str) -> Optional[str]:
        if not content:
            return None
            
        # Look for code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', content, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
            
        return content.strip()

    def test_kernel_compilation(self, code: str) -> tuple[bool, str, float]:
        """TODO: needs to be ported from benchmark maybe?"""
        try:
            temp_module = {}
            exec(code, temp_module)
            
            if 'ModelNew' not in temp_module:
                return False, "ModelNew class not found in generated code", float('inf')
            
            model = temp_module['ModelNew']()
            model = model.cuda()
            
            test_input = torch.randn(32, 128).cuda()
            
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                output = model(test_input)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            
            return True, "Compilation and execution successful", elapsed_time
            
        except Exception as e:
            return False, f"Compilation/execution error: {str(e)}", float('inf')

    def generate_kernel_single_turn(self, problem_description: str, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model="cognition-ai/Kevin-32B",
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in API call: {e}")
            return ""

    def generate_kernel(self, problem_description: str, turns: int = 3) -> str:
        """Generate CUDA kernel using multi-turn
        
        Args:
            problem_description: The original kernel description/code
            turns: Number of optimization turns
            
        Returns:
            Final optimized kernel code
        """
        print(f"Starting Kevin kernel generation with {turns} turns...")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Here is the architecture to optimize:\n\n{problem_description}"}
        ]
        
        best_kernel = ""
        best_time = float('inf')
        
        for turn in range(turns):
            print(f"Turn {turn + 1}/{turns}...")
            
            response_content = self.generate_kernel_single_turn(problem_description, messages)
            
            if not response_content:
                print(f"Turn {turn + 1}: No response received")
                continue
                
            kernel_code = self.extract_code_from_response(response_content)
            
            if not kernel_code:
                print(f"Turn {turn + 1}: No code extracted from response")
                continue
            
            print(f"Turn {turn + 1}: Generated kernel ({len(kernel_code)} chars)")
            
            compiles, feedback, exec_time = self.test_kernel_compilation(kernel_code)
            
            if compiles:
                print(f"Turn {turn + 1}: Compilation successful, execution time: {exec_time:.3f}ms")
                if exec_time < best_time:
                    best_kernel = kernel_code
                    best_time = exec_time
                    print(f"Turn {turn + 1}: New best kernel found!")
                
                if turn < turns - 1:
                    messages.append({
                        "role": "assistant", 
                        "content": response_content
                    })
                    messages.append({
                        "role": "user", 
                        "content": f"Great! The kernel compiled and ran successfully with execution time {exec_time:.3f}ms. Can you optimize it further for even better performance? Focus on memory access patterns, shared memory usage, and kernel fusion opportunities."
                    })
            else:
                print(f"Turn {turn + 1}: Compilation failed - {feedback}")
                
                if turn < turns - 1:
                    messages.append({
                        "role": "assistant", 
                        "content": response_content
                    })
                    messages.append({
                        "role": "user", 
                        "content": f"The kernel failed to compile with error: {feedback}. Please fix the compilation issues and ensure the code is syntactically correct."
                    })
        
        if best_kernel:
            print(f"Best kernel found with execution time: {best_time:.3f}ms")
            return best_kernel
        else:
            print("No successful kernel generated, returning last attempt")
            return kernel_code if 'kernel_code' in locals() else ""


_generator = None

def get_generator():
    global _generator
    if _generator is None:
        _generator = KevinKernelGenerator()
    return _generator

def generate_kernel(problem_description: str, turns: int = 3) -> str:
    """Main entry point for CUDA kernel generation
    
    Args:
        problem_description: The kernel problem description/code to optimize
        turns: Number of optimization turns (default: 3)
        
    Returns:
        Optimized CUDA kernel code
    """
    generator = get_generator()
    return generator.generate_kernel(problem_description, turns) 