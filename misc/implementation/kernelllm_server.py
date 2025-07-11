import argparse
import subprocess
import sys
import os
from pathlib import Path

"""facebook/KernelLLM 8B model"""

def run_sglang_server(args):
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", "facebook/KernelLLM",
        
        "--tp-size", "1", 
    ]
    
    if args.port:
        cmd.extend(["--port", str(args.port)])
    
    if args.host:
        cmd.extend(["--host", args.host])
    
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    
    if args.max_total_tokens:
        cmd.extend(["--max-total-tokens", str(args.max_total_tokens)])
    
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    env = os.environ.copy()
    if not env.get("TORCHINDUCTOR_CACHE_DIR"):
        env["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/sglang_torch_cache"
    
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print(f"Setting CUDA_VISIBLE_DEVICES={args.device}")
    
    print("Launching KernelLLM-8B...")
    print("Configuration:")
    print(f"Model: facebook/KernelLLM")
    print(f"Size: 8B parameters")
    print(f"Quantization: None (native fp16/bf16)")
    print(f"Tensor Parallelism: 1 (single GPU)")
    print(f"GPU Device: {args.device if args.device is not None else 'default'}")
    print(f"Port: {args.port or 30000}")
    print()
    
    print("Server will be available at: http://{}:{}".format(args.host or "localhost", args.port or 30000))
    print()
    
    if args.dry_run:
        print("Dry run - command that would be executed:")
        print(" ".join(cmd))
        print(f"Environment: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        return
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching SGLang server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(
        description="Launch SGLang server with facebook/KernelLLM-8B model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic launch on default GPU
  python kernelllm_server.py

  # Launch on GPU 1
  python kernelllm_server.py --device 1

  # Launch on GPU 0 with custom port
  python kernelllm_server.py --device 0 --port 8080 --api-key your-secret-key

  # Enable metrics and debug logging on GPU 1
  python kernelllm_server.py --device 1 --enable-metrics --log-level debug

  # Dry run to see the command
  python kernelllm_server.py --device 1 --dry-run
        """
    )
    
    parser.add_argument("--device", type=int,
                       help="GPU device ID to use. If not specified, uses default device.")
    
    parser.add_argument("--port", type=int, default=30000,
                       help="Port to run the server on (default: 30000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--api-key", type=str,
                       help="API key for server authentication")
    
    parser.add_argument("--max-total-tokens", type=int,
                       help="Maximum total tokens in memory pool (default: 32768)")
    
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code from HuggingFace")
    
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"],
                       default="info", help="Logging level (default: info)")
    parser.add_argument("--enable-metrics", action="store_true",
                       help="Enable Prometheus metrics")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Show the command that would be executed without running it")
    
    args = parser.parse_args()
    
    run_sglang_server(args)

if __name__ == "__main__":
    main()