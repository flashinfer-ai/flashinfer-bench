import argparse
import subprocess
import sys
import os
from pathlib import Path

"""Optimized for dual 4090 on 8180 Catalyst, TODO: results are poor, 48GB is not enough"""

def run_sglang_server(args):
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", "cognition-ai/Kevin-32B",
        
        #tensor parallelism
        "--tp-size", "2", 
        
        "--mem-fraction-static", "0.85",
        "--disable-cuda-graph-padding",
        "--max-total-tokens", "16384",
        
        "--schedule-policy", "fcfs",
        "--stream-interval", "1",
    ]
    
    # Add disable-cuda-graph if requested to fix libstdc++ compatibility issues
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    
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
    
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])

    env = os.environ.copy()
    if not env.get("TORCHINDUCTOR_CACHE_DIR"):
        env["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/sglang_torch_cache"
    
    # Fix libstdc++ compatibility by using system library if requested
    if args.use_system_libstdcxx:
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        system_lib_path = "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
        if current_ld_path:
            env["LD_LIBRARY_PATH"] = f"{system_lib_path}:{current_ld_path}"
        else:
            env["LD_LIBRARY_PATH"] = system_lib_path
        print(f"Using system libstdc++ for GLIBCXX_3.4.32+ compatibility")
    
    if args.devices is not None:
        if isinstance(args.devices, list):
            devices_str = ",".join(map(str, args.devices))
        else:
            devices_str = str(args.devices)
        env["CUDA_VISIBLE_DEVICES"] = devices_str
        print(f"Setting CUDA_VISIBLE_DEVICES={devices_str}")
    
    print("Launching Kevin-32B...")
    print("Configuration:")
    print(f"Model: cognition-ai/Kevin-32B")
    print(f"Quantization: W8A8-INT8 (memory-optimized)")
    print(f"KV Cache: FP8-E5M2")
    print(f"Tensor Parallelism: 2 GPUs")
    print(f"GPU Devices: {args.devices if args.devices is not None else 'default'}")
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
        description="Launch SGLang server with Cognition AI Kevin-32B model optimized for dual RTX 4090 setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic launch
  python kevin_server.py

  # Launch on specific GPUs (0,1)
  python kevin_server.py --devices 0,1

  # Launch on GPUs 2,3 with custom port
  python kevin_server.py --devices 2,3 --port 8080 --api-key your-secret-key

  # Fix libstdc++ compatibility issues (preferred solution)
  python kevin_server.py --devices 0,1 --use-system-libstdcxx

  # Alternative: disable CUDA graphs (slower but works)
  python kevin_server.py --devices 0,1 --disable-cuda-graph

  # Reduce memory usage if running into OOM
  python kevin_server.py --devices 0,1 --mem-fraction 0.75 --chunked-prefill 2048

  # Enable metrics and debug logging
  python kevin_server.py --devices 0,1 --enable-metrics --log-level debug

  # Dry run to see the command
  python kevin_server.py --devices 0,1 --dry-run
        """
    )
    
    parser.add_argument("--devices", type=str,
                       help="Comma-separated list of GPU device IDs to use (e.g., '0,1'). If not specified, uses default devices.")
    
    parser.add_argument("--port", type=int, default=30000,
                       help="Port to run the server on (default: 30000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--api-key", type=str,
                       help="API key for server authentication")
    
    
    parser.add_argument("--max-total-tokens", type=int,
                       help="Maximum total tokens in memory pool")
    
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code from HuggingFace")
    
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"],
                       default="info", help="Logging level (default: info)")
    parser.add_argument("--enable-metrics", action="store_true",
                       help="Enable Prometheus metrics")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Show the command that would be executed without running it")
    
    parser.add_argument("--dtype", type=str,
                       help="Data type for the model")
    
    parser.add_argument("--disable-cuda-graph", action="store_true",
                       help="Disable CUDA graphs to work around libstdc++ compatibility issues")
    
    parser.add_argument("--use-system-libstdcxx", action="store_true",
                       help="Use system libstdc++ for GLIBCXX_3.4.32+ compatibility")
    
    args = parser.parse_args()
    
    # Parse devices string into list if provided
    if args.devices:
        args.devices = args.devices.split(',')
    
    run_sglang_server(args)

if __name__ == "__main__":
    main()