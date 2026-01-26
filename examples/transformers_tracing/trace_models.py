#!/usr/bin/env python3
"""
Trace transformers models to collect flashinfer-bench workload traces.

This script demonstrates how to trace various LLM models using the
flashinfer-bench transformers integration. It supports:
- Qwen3-30B-A3B (MoE model)
- LLaMA-3.1-70B
- GPT-OSS-120B

Usage:
    # Trace a specific model
    python trace_models.py --model meta-llama/Llama-3.1-70B-Instruct --output ./traces

    # Trace with FP8 quantization
    python trace_models.py --model RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 --output ./traces

    # Use environment variables (alternative)
    FIB_ENABLE_TRACING=1 FIB_DATASET_PATH=./traces python your_script.py

Requirements:
    - transformers
    - torch
    - flashinfer-bench
    - accelerate (for large models)
    - bitsandbytes or auto-gptq (for quantization)
"""

import argparse
import os
import sys
from pathlib import Path

import torch


def setup_tracing(output_path: str):
    """Setup flashinfer-bench tracing."""
    from flashinfer_bench.tracing import enable_tracing
    
    # Create output directory if needed
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    return enable_tracing(output_path)


def trace_model(
    model_id: str,
    output_path: str,
    prompts: list[str] | None = None,
    max_new_tokens: int = 50,
    device: str = "auto",
    torch_dtype: str = "auto",
):
    """
    Trace a transformers model and collect workload traces.
    
    Parameters
    ----------
    model_id : str
        HuggingFace model ID (e.g., "meta-llama/Llama-3.1-70B-Instruct")
    output_path : str
        Path to save traces
    prompts : list[str], optional
        List of prompts to run inference on. Defaults to sample prompts.
    max_new_tokens : int
        Maximum tokens to generate per prompt
    device : str
        Device to use ("auto", "cuda", "cpu")
    torch_dtype : str
        Torch dtype ("auto", "float16", "bfloat16")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "Hello, how are you today?",
            "Explain the concept of machine learning in simple terms.",
            "Write a short poem about artificial intelligence.",
            "What are the key differences between Python and JavaScript?",
            "Summarize the main benefits of renewable energy.",
        ]
    
    print(f"Loading model: {model_id}")
    print(f"Output path: {output_path}")
    print(f"Device: {device}")
    print(f"Torch dtype: {torch_dtype}")
    print("-" * 50)
    
    # Determine torch dtype
    if torch_dtype == "auto":
        dtype = "auto"
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = "auto"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device,
        "trust_remote_code": True,
    }
    
    # Enable tracing and load model
    with setup_tracing(output_path):
        print("Loading model (this may take a while for large models)...")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        print(f"Model loaded. Running inference on {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt[:50]}...")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate (this triggers all the operations we want to trace)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode and print result
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   Generated: {generated_text[:100]}...")
        
        print("\n" + "=" * 50)
        print("Tracing complete! Traces will be flushed to disk.")
    
    print(f"\nTraces saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Trace transformers models with flashinfer-bench"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID to trace",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./traces",
        help="Output directory for traces",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16"],
        help="Torch dtype to use",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom prompts to use (space-separated)",
    )
    
    args = parser.parse_args()
    
    trace_model(
        model_id=args.model,
        output_path=args.output,
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=args.dtype,
    )


# Predefined model configurations for the target models
MODEL_CONFIGS = {
    "qwen3-30b-moe": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "description": "Qwen3 30B MoE model (3B active parameters)",
        "expected_ops": ["attention", "rmsnorm", "rope", "embedding", "silu", "moe", "linear", "softmax"],
    },
    "qwen3-30b-moe-fp8": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "description": "Qwen3 30B MoE model with FP8 quantization",
        "expected_ops": ["attention", "rmsnorm", "rope", "embedding", "silu", "moe", "linear", "softmax"],
    },
    "llama-3.1-70b": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "description": "LLaMA 3.1 70B Instruct model",
        "expected_ops": ["attention", "rmsnorm", "rope", "embedding", "silu", "linear", "softmax"],
    },
    "llama-3.1-70b-fp8": {
        "model_id": "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8",
        "description": "LLaMA 3.1 70B with FP8 quantization",
        "expected_ops": ["attention", "rmsnorm", "rope", "embedding", "silu", "linear", "softmax"],
    },
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b",
        "description": "GPT-OSS 120B MoE model",
        "expected_ops": ["attention", "rmsnorm", "rope", "embedding", "gelu", "moe", "linear", "softmax"],
    },
}


if __name__ == "__main__":
    main()
