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
        
        #int8 and fp8 quantization
        "--quantization", "fp8",
        "--kv-cache-dtype", "fp8",
        
        "--mem-fraction-static", "0.85",
        "--disable-cuda-graph-padding",
        "--max-total-tokens", "16384",
        
        "--schedule-policy", "fcfs",
        "--stream-interval", "1",
    ]
    
    if args.port:
        cmd.extend(["--port", str(args.port)])
    
    if args.host:
        cmd.extend(["--host", args.host])
    

    env = os.environ.copy()
    if not env.get("TORCHINDUCTOR_CACHE_DIR"):
        env["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/sglang_torch_cache"
    
    print("Launching Kevin-32B...")
    print("Configuration:")
    print(f"Model: cognition-ai/Kevin-32B")
    print(f"Quantization: W8A8-INT8 (memory-optimized)")
    print(f"KV Cache: FP8-E5M2")
    print(f"Port: {args.port or 30000}")
    print()
    
    print("Server will be available at: http://{}:{}".format(args.host or "localhost", args.port or 30000))
    print()
    
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
  python sglang_kevin_launcher.py

  # Custom port and API key
  python sglang_kevin_launcher.py --port 8080 --api-key your-secret-key

  # Reduce memory usage if running into OOM
  python sglang_kevin_launcher.py --mem-fraction 0.75 --chunked-prefill 2048

  # Enable metrics and debug logging
  python sglang_kevin_launcher.py --enable-metrics --log-level debug

  # Dry run to see the command
  python sglang_kevin_launcher.py --dry-run
        """
    )
    
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
    
    args = parser.parse_args()
    
    run_sglang_server(args)

if __name__ == "__main__":
    main()