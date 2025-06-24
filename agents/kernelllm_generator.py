"""adapted from kernelllm.py on the hugginface repo: https://huggingface.co/facebook/KernelLLM/blob/main/kernelllm.py
Uses sglang to host the model"""

#Important Note, for Catalyst 8180, host the model on device 1, the model is SFTed to use device 0 only in codegen, will have memory issues during benchmarking

import openai

DEFAULT_MODEL_CODE = """
import torch
import torch.nn as nn
class Model(nn.Module):
    \"\"\"
    A model that computes Hinge Loss for binary classification tasks.
    Parameters:
        None
    \"\"\"
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))
batch_size = 128
input_shape = (1,)
dim = 1
def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]
def get_init_inputs():
    return []
"""

PROMPT_TEMPLATE = """
<|begin_of_text|>You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.
You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.
Here's an example to show you the syntax of inline embedding custom operators from the Triton DSL in torch: The example given architecture is:
```
import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, a, b):
        return a + b
def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]
def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
```
The example new arch with custom Triton kernels looks like this:
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)
def triton_add(x: torch.Tensor, y: torch.Tensor):
    \"\"\"
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    \"\"\"
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)
```
You are given the following architecture:
```
{}
```
Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
"""


class KernelLLM:
    def __init__(
        self,
        port: int = 30000,
        host: str = "127.0.0.1",
        model_name: str = "cognition-ai/Kevin-32B"
    ):
        self.client = openai.Client(
            base_url=f"http://{host}:{port}/v1", 
            api_key="None"
        )
        self.model_name = model_name

    def generate_raw(
        self, prompt: str, temperature: float = 0.6, max_new_tokens: int = 2048
    ) -> str:
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in API call: {e}")
            return ""

    def generate_kernel(
        self, code: str, temperature: float = 0.6, max_new_tokens: int = 2048
    ) -> str:
        """
        Generate Triton for the given torch module.
        The input code should be a python module that contains a torch Model(nn.Module) class and
        `get_inputs()` and `get_init_inputs()` functions such that your model can be run like this
            ```
            args, kwargs = get_inputs()
            model = Model(*args, **kwargs)
            out = model(get_inputs())
            ```
        Args:
            code (str): The torch code to generate Triton for.
            temperature (float): The temperature to use for sampling.
            max_new_tokens (int): The maximum length of the generated text.
        Returns:
            str: The generated Triton module.
        """
        prompt = PROMPT_TEMPLATE.format(code)
        return self.generate_raw(prompt, temperature, max_new_tokens)

    def generate_triton(self, code: str, temperature: float = 0.6, max_new_tokens: int = 2048) -> str:
        return self.generate_kernel(code, temperature, max_new_tokens)


_generator = None

def get_generator():
    global _generator
    if _generator is None:
        _generator = KernelLLM()
    return _generator

def generate_kernel(problem_description: str, turns: int = 1) -> str:
    """Main entry point for KernelLLM Triton kernel generation
    
    Args:
        problem_description: The PyTorch model code to optimize
        turns: Number of kernel candidates to generate (ignored)
        
    Returns:
        Generated Triton kernel code
    """
    generator = get_generator()
    return generator.generate_kernel(problem_description) 