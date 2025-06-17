### Running Llama3.1-8B single GPU would run the following kernels

A single forward would run the following kernels (essentially the kernels in the architecture of `LlamaForCausalLM`)

- LlamaModel()
    - VocabParallelEmbedding()
        - **Kernel: `F.embedding`**
    - 32 LlamaDecoderLayer
        - RMSNorm (input_layernorm)
            - **Kernel: first layer is `rmsnorm`, subsequent are `fused_add_rmsnorm`**
        - LlamaAttention
            - QKVParallelLinear
                - **Kernel: `F.linear`**
            - Llama3RotaryEmbedding
                - **Kernel: `apply_rope_with_cos_sin_cache_inplace`**
            - RadixAttention (simplified below)
                - If prefill:
                    - **Kernel: `flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.forward()`**
                - Else decode
                    - **Kernel: `flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.forward()`**
            - RowParallelLinear
                - **Kernel: `F.linear`**
        - RMSNorm (post_attention_layernorm)
            - **Kernel: `fused_add_rmsnorm`**
        - LlamaMLP
            - MergedColumnParallelLinear
                - **Kernel: `F.linear`**
            - SiluAndMul
                - **Kernel: `act_and_mul_kernel`**
            - RowParallelLinear
                - **Kernel: `F.linear`**
    - RMSNorm
        - **Kernel: `fused_add_rmsnorm`**
- logits_processor()
    - **Kernel: `torch.matmul`**

Note that we ignore the `F.` operators and `torch.` operators.

This leaves us with the following kernels:
- `rmsnorm` and `fused_add_rmsnorm`
- `act_and_mul_kernel` (Silu specifically, for the activation)
- `apply_rope_with_cos_sin_cache_inplace`
- `BatchPrefillWithRaggedKVCacheWrapper`
- `BatchDecodeWithPagedKVCacheWrapper`

### For each kernel, we implement a file with
- `forward_flashinfer()`: implementation of the operator by dispatching to FlashInfer's kernel
- `forward_pytorch()`: a naive implementation of the operator with PyTorch
- `operator_description()`: information about the operator (e.g. shapes, computation, etc.) with the goal of aiding the LLM to generate
