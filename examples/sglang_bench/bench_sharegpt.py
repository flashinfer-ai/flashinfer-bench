"""
Benchmark SGLang serving throughput using the ShareGPT dataset.

Launches an SGLang server, sends batched requests from ShareGPT, and reports
throughput and latency statistics across a sweep of batch sizes. Model server
flags are loaded from model_configs.json, keyed by GPU type and model name.

Usage:
    # Auto-detect GPU type from nvidia-smi
    python3 bench_sharegpt.py --model llama-3.1-8b  --model-path /path/to/Llama-3.1-8B-Instruct
    python3 bench_sharegpt.py --model deepseek-v3   --model-path /path/to/DeepSeek-V3
    python3 bench_sharegpt.py --model qwen3-235b-a22b --model-path /path/to/Qwen3-235B-A22B

    # Explicitly specify GPU type (b200, h200, h100, mi300x)
    python3 bench_sharegpt.py --gpu h100 --model llama-3.1-70b --model-path /path/to/Llama-3.1-70B-Instruct

    # TP size is auto-detected from CUDA_VISIBLE_DEVICES (set by gpu-lock):
    #   tools/gpu-lock --gpus 4 -- python3 bench_sharegpt.py --model qwen3-235b-a22b ...
    #   → CUDA_VISIBLE_DEVICES=0,1,2,3 → TP=4 used automatically

    # Multi-node mode (auto-detected when config TP > local GPU count):
    #   Peer nodes are discovered from SLURM_JOB_ID automatically.
    #   Workers are launched via SSH; passwordless SSH between nodes is required.
    #   Manual override: --peer-node-addr nvl72155-T14
    #   Disable auto multi-node: --no-multinode


    # Tracing flags
    python3 bench_sharegpt.py --model llama-3.1-8b --model-path /path/to/model --disable-radix-cache
    python3 bench_sharegpt.py --model qwen3-235b-a22b --model-path /path/to/model --enable-deterministic-inference

    # Custom batch sizes and number of batches
    python3 bench_sharegpt.py --model llama-3.1-8b --model-path /path/to/model --batch-sizes 32 128 --num-batches 8

    # Workload collection: restart server per batch size so each gets its own DUMP_MAX_COUNT budget
    FLASHINFER_DUMP_MAX_COUNT=500 FLASHINFER_DUMP_INCLUDE="BatchDecodeWithPagedKVCacheWrapper*" \
    FLASHINFER_DUMP_EXCLUDE="*.__init__" \
    python3 bench_sharegpt.py --model llama-4-scout-ps64 --model-path /path/to/model \
        --batch-sizes 64 128 --num-batches 4 --restart-per-batch-size --disable-cuda-graph

Environment variables (optional, for flashinfer-bench tracing):
    FIB_ENABLE_APPLY=1        Enable the flashinfer-bench apply hook
    FIB_DATASET_PATH=<dir>    Directory to write flashinfer trace data

    Example:
        FIB_ENABLE_APPLY=1 FIB_DATASET_PATH=/path/to/traces/ python3 bench_sharegpt.py --model llama-3.1-8b --model-path /path/to/model
"""

import argparse
import asyncio
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

_CONFIG_FILE = Path(__file__).parent / "model_configs.json"

# Compute nvidia CUDA header paths once at startup.
# pip-installed nvidia packages (nvidia-cuda-nvrtc-cu13 etc.) place nvrtc.h,
# cuda_runtime.h etc. under site-packages/nvidia/cuXX/include.  These are not
# on the default include path of the system compiler, but JIT kernels such as
# flashinfer's TRT-LLM FMHA module need them.  Exporting CPATH makes them
# visible to every subprocess (SGLang server, SSH peer worker) without patching
# individual compile commands.
_NVIDIA_INCLUDES: List[str] = sorted(
    str(p) for p in Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cu*/include")
)
# Also include CUDA CTK headers from conda env's targets/ directory (e.g.
# targets/sbsa-linux/include on ARM, targets/x86_64-linux/include on x86).
# These provide <nv/target> and other CTK-private headers required by the
# FlashInfer TRT-LLM JIT compilation.
_NVIDIA_INCLUDES += sorted(
    str(p) for p in Path(sys.prefix).glob("targets/*/include") if (p / "nv" / "target").exists()
)
_NVIDIA_CPATH: str = ":".join(_NVIDIA_INCLUDES)

# Map nvidia-smi GPU name substrings → sgl-cookbook hardware IDs
_GPU_NAME_MAP = [
    ("B200", "b200"),
    ("H200", "h200"),
    ("H100", "h100"),
    ("A100", "a100"),
    ("MI355", "mi355x"),
    ("MI325", "mi325x"),
    ("MI300", "mi300x"),
]


def detect_gpu_type() -> str:
    """Auto-detect GPU type from nvidia-smi or rocm-smi, returning a sgl-cookbook hardware ID."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            name = result.stdout.strip().splitlines()[0].upper()
            for substr, gpu_id in _GPU_NAME_MAP:
                if substr in name:
                    return gpu_id
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            name = result.stdout.upper()
            for substr, gpu_id in _GPU_NAME_MAP:
                if substr in name:
                    return gpu_id
    except Exception:
        pass
    return "b200"  # default fallback


def detect_tp_size() -> Optional[int]:
    """Detect available GPU count from CUDA_VISIBLE_DEVICES (set by gpu-lock) to use as TP size."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd != "-1":
        return len(cvd.split(","))
    # Fall back to total GPU count from nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            count = len(result.stdout.strip().splitlines())
            return count if count > 0 else None
    except Exception:
        pass
    return None


def get_slurm_peer_nodes() -> List[Tuple[str, str]]:
    """
    Return [(hostname, ip), ...] for peer nodes in the current SLURM job allocation,
    excluding this node. Returns an empty list when not in a multi-node SLURM job.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        return []
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id], capture_output=True, text=True, timeout=10
        )
        node_list = None
        for line in result.stdout.splitlines():
            for field in line.split():
                if field.startswith("NodeList=") and "(null)" not in field:
                    node_list = field.split("=", 1)[1]
                    break
        if not node_list:
            return []

        result = subprocess.run(
            ["scontrol", "show", "hostnames", node_list], capture_output=True, text=True, timeout=10
        )
        all_nodes = [n for n in result.stdout.strip().splitlines() if n]
        current_host = socket.gethostname().split(".")[0]

        peers = []
        for node in all_nodes:
            if node.split(".")[0] == current_host:
                continue
            try:
                ip = socket.getaddrinfo(node, None)[0][4][0]
            except Exception:
                ip = node
            peers.append((node, ip))
        return peers
    except Exception:
        return []


def get_head_node_ip_for_peer(peer_ip: str) -> str:
    """Return the local IP address that routes to peer_ip (used as --dist-init-addr host)."""
    try:
        result = subprocess.run(
            ["ip", "route", "get", peer_ip], capture_output=True, text=True, timeout=5
        )
        tokens = result.stdout.split()
        for i, tok in enumerate(tokens):
            if tok == "src" and i + 1 < len(tokens):
                return tokens[i + 1]
    except Exception:
        pass
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5)
        ips = result.stdout.strip().split()
        if ips:
            return ips[0]
    except Exception:
        pass
    return socket.gethostname()


def launch_peer_worker(
    peer_host: str,
    model_path: str,
    server_args: List[str],
    head_ip: str,
    dist_port: int,
    total_nodes: int,
    node_rank: int,
    conda_env: Optional[str],
) -> subprocess.Popen:
    """
    SSH to peer_host and start sglang.launch_server as a TP worker.

    FLASHINFER_* dump env vars are intentionally NOT forwarded — tensor dumps
    are only collected from rank 0 (the head node).
    """
    worker_args = [
        "--model-path",
        model_path,
        "--nnodes",
        str(total_nodes),
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        f"{head_ip}:{dist_port}",
    ] + server_args

    # Use sys.executable so the remote worker runs in the same Python environment
    # as the head node.  This works because /home is NFS-shared across nodes.
    # Fall back to "conda run -n <env>" only when sys.executable is unavailable
    # (e.g. dry-run testing or a non-NFS cluster).
    python_bin = sys.executable
    python_cmd = [python_bin, "-m", "sglang.launch_server"] + worker_args
    cmd_str = " ".join(shlex.quote(c) for c in python_cmd)

    if conda_env and not os.path.isabs(python_bin):
        # Fallback: use conda run when sys.executable is a relative path
        conda_bin = shutil.which("conda") or "conda"
        remote_cmd = f"{conda_bin} run --no-capture-output -n {shlex.quote(conda_env)} {cmd_str}"
    else:
        # SSH sessions don't inherit the conda env's CUDA_HOME or PATH.
        # Use `env` to set both so the command works in any remote shell
        # (csh/tcsh don't support the `KEY=value cmd` inline-assignment syntax;
        # `env` is an external binary that works universally).
        # CUDA_HOME: deep_gemm checks it first; conda prefix contains nvcc/include/lib.
        # PATH: conda env's bin is prepended so JIT tools (ninja, nvcc) are found.
        # CPATH: pip-installed nvidia packages put nvrtc.h etc. under
        #   site-packages/nvidia/cuXX/include; add them so TRT-LLM JIT can find them.
        #   We use the module-level _NVIDIA_CPATH constant (computed from sys.prefix on
        #   the head node; the path is identical on the peer node via NFS).
        # FLASHINFER_WORKSPACE_BASE: point peer node JIT cache to local /tmp so the
        #   peer doesn't share the NFS lock files with the head node, which causes
        #   OSError: [Errno 116] Stale file handle during FileLock acquisition.
        conda_bin_dir = os.path.dirname(python_bin)  # .../envs/XXX/bin
        conda_prefix = os.path.dirname(conda_bin_dir)  # .../envs/XXX
        base_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        env_vars = (
            f"CUDA_HOME={shlex.quote(conda_prefix)}"
            f" PATH={shlex.quote(conda_bin_dir)}:{base_path}"
            f" FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_jit_peer"
            f" FLASHINFER_CUBIN_DIR=/tmp/flashinfer_cubins"
            f" SGLANG_ENABLE_JIT_DEEPGEMM=0"
            f" TRITON_CACHE_DIR=/tmp/triton_cache_peer"
        )
        if _NVIDIA_CPATH:
            env_vars += f" CPATH={shlex.quote(_NVIDIA_CPATH)}"
        remote_cmd = f"env {env_vars} {cmd_str}"

    log_file = open(f"/tmp/sglang_worker_rank{node_rank}_{peer_host}.log", "w")
    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        peer_host,
        remote_cmd,
    ]
    log(f"Launching peer worker (rank {node_rank}) on {peer_host} (log: {log_file.name})")
    return subprocess.Popen(ssh_cmd, stdout=log_file, stderr=log_file)


from datasets import load_dataset
from sglang.bench_serving import benchmark, set_global_args
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)


@dataclass
class TestRequest:
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None
    extra_request_body: Dict[str, Any] = field(default_factory=dict)
    routing_key: Optional[str] = None


def load_model_configs() -> dict:
    """Load model configurations from model_configs.json."""
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(f"Model config file not found: {_CONFIG_FILE}")
    with _CONFIG_FILE.open() as f:
        data = json.load(f)
    # Strip metadata keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def get_available_models(gpu_type: str) -> List[str]:
    """Return model keys available for a given GPU type."""
    configs = load_model_configs()
    gpu_key = gpu_type.lower()
    if gpu_key not in configs:
        # Fall back to first GPU type defined
        gpu_key = next(iter(configs))
    return list(configs[gpu_key].keys())


def get_model_config(model_type: str, gpu_type: str) -> dict:
    """Return server flags for a given model + GPU type from model_configs.json."""
    configs = load_model_configs()
    gpu_key = gpu_type.lower()
    if gpu_key not in configs:
        available_gpus = list(configs.keys())
        raise ValueError(f"Unknown GPU type {gpu_type!r}. Available: {available_gpus}")
    model_key = model_type.lower()
    gpu_configs = configs[gpu_key]
    if model_key not in gpu_configs:
        available_models = list(gpu_configs.keys())
        raise ValueError(
            f"Model {model_type!r} not configured for GPU {gpu_type!r}. "
            f"Available models for {gpu_type}: {available_models}"
        )
    return gpu_configs[model_key]


def log(msg: str) -> None:
    print(f"[BENCHMARK] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


def load_prompts_from_sharegpt(n: int) -> List[str]:
    """Load n prompts from the ShareGPT dataset."""
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
        streaming=True,
    )
    prompts = []
    for example in ds:
        conv = example.get("conversations", [])
        if not conv:
            continue
        # Dataset returns conversations as JSON strings in newer datasets library versions
        first = conv[0]
        if isinstance(first, str):
            import json as _json

            try:
                first = _json.loads(first)
            except Exception:
                continue
        if isinstance(first, dict) and first.get("from", "").lower() == "human":
            prompts.append(first.get("value", ""))
        if len(prompts) >= n:
            break

    log(f"Loaded {len(prompts)} prompts from ShareGPT dataset")
    prompt_lengths = [len(p) for p in prompts]
    if prompt_lengths:
        log(
            f"Prompt length statistics — "
            f"Min: {min(prompt_lengths)}, "
            f"Max: {max(prompt_lengths)}, "
            f"Avg: {sum(prompt_lengths) / len(prompt_lengths):.1f}"
        )
    return prompts


class DummyTokenizer:
    """Minimal tokenizer shim required by the benchmark function."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return []


def build_bench_args() -> SimpleNamespace:
    """Build the SimpleNamespace expected by bench_serving.benchmark."""
    return SimpleNamespace(
        backend="sglang",
        dataset_name="custom",
        disable_ignore_eos=False,
        disable_stream=True,
        return_logprob=False,
        return_routed_experts=False,
        output_file=None,
        output_details=False,
        warmup_requests=0,
        plot_throughput=False,
        header=None,
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        profile_activities=["CPU", "GPU"],
        profile_num_steps=None,
        profile_by_stage=False,
        profile_stages=None,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
    )


def run_benchmark(
    base_url: str,
    prompts: List[str],
    batch_size: int,
    temperature: float = 0.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> list:
    """Run the benchmark over prompts in batches and return all results."""
    tokenizer = DummyTokenizer()
    set_global_args(build_bench_args())

    num_batches = math.ceil(len(prompts) / batch_size)
    all_results = []

    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        current_time = time.time()
        input_requests = [
            TestRequest(
                prompt=p,
                prompt_len=0,
                output_len=0,
                timestamp=current_time,
                text_prompt_len=0,
                vision_prompt_len=0,
            )
            for p in batch_prompts
        ]

        log(f"Running batch {i + 1}/{num_batches} with {len(input_requests)} prompts")
        results = asyncio.run(
            benchmark(
                backend="sglang",
                api_url=f"{base_url}/generate",
                base_url=base_url,
                model_id="default",
                tokenizer=tokenizer,
                input_requests=input_requests,
                request_rate=float("inf"),
                max_concurrency=batch_size,
                disable_tqdm=False,
                lora_names=None,
                lora_request_distribution=None,
                lora_zipf_alpha=None,
                extra_request_body={
                    "sampling_params": {
                        "temperature": temperature,
                        **({"top_k": top_k} if top_k > 0 else {}),
                        **({"top_p": top_p} if top_p < 1.0 else {}),
                    }
                },
                profile=False,
            )
        )
        all_results.append(results)

    return all_results


def _shutdown_server(process: subprocess.Popen, peer_processes: List[subprocess.Popen]) -> None:
    """Kill the SGLang head server and all peer workers."""
    kill_process_tree(process.pid)
    for p in peer_processes:
        try:
            peer_host = p.args[5]  # ssh cmd: ["ssh", "-o", OPT, "-o", OPT, HOST, remote_cmd]
            subprocess.run(
                [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "BatchMode=yes",
                    peer_host,
                    "pkill -f 'sglang.launch_server'",
                ],
                timeout=10,
            )
        except Exception:
            pass
        try:
            kill_process_tree(p.pid)
        except Exception:
            pass
    time.sleep(3)
    log("Server shutdown complete")


def _launch_server_session(
    model_path: str,
    base_url: str,
    server_args: List[str],
    server_env: Dict[str, str],
    need_multinode: bool,
    peers: List[Tuple[str, str]],
    nnodes: int,
    head_ip: Optional[str],
    dist_addr: Optional[str],
    dist_port: int,
    conda_env: str,
    timeout: int,
) -> Tuple[subprocess.Popen, List[subprocess.Popen]]:
    """Launch peer workers (if multi-node) and the SGLang head server.

    Returns ``(head_process, peer_processes)``.  The caller is responsible for
    shutting the session down via :func:`_shutdown_server` when done.
    """
    peer_processes: List[subprocess.Popen] = []
    launch_args = list(server_args)

    if need_multinode:
        for rank, (peer_host, _) in enumerate(peers, start=1):
            p = launch_peer_worker(
                peer_host=peer_host,
                model_path=model_path,
                server_args=server_args,
                head_ip=head_ip,
                dist_port=dist_port,
                total_nodes=nnodes,
                node_rank=rank,
                conda_env=conda_env,
            )
            peer_processes.append(p)

        launch_args = [
            "--nnodes",
            str(nnodes),
            "--node-rank",
            "0",
            "--dist-init-addr",
            dist_addr,
        ] + launch_args

    process = popen_launch_server(
        model_path, base_url, timeout=timeout, other_args=launch_args, env=server_env
    )
    return process, peer_processes


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ShareGPT dataset with SGLang using different models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU type (b200, h200, h100, mi300x). Auto-detected from nvidia-smi if not specified.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key to use from model_configs.json (e.g. llama-3.1-8b, qwen3-235b-a22b).",
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model weights directory"
    )
    parser.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Add --disable-radix-cache to server args (for ragged prefill collection)",
    )
    parser.add_argument(
        "--enable-deterministic-inference",
        action="store_true",
        help="Add --enable-deterministic-inference to server args (for paged prefill collection)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:20000",
        help="Base URL of the SGLang server (default: http://127.0.0.1:20000)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 16, 64],
        help="List of batch sizes to benchmark (default: 1 16 64)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Number of batches per batch-size sweep (default: 4)",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Pass --disable-cuda-graph to the SGLang server",
    )
    parser.add_argument(
        "--peer-node-addr",
        type=str,
        nargs="+",
        default=None,
        metavar="HOST",
        help=(
            "Peer node hostname(s)/IP(s) for multi-node TP. "
            "Auto-detected from SLURM_JOB_ID when not specified."
        ),
    )
    parser.add_argument(
        "--dist-init-port",
        type=int,
        default=20010,
        help="Port for PyTorch distributed rendezvous (--dist-init-addr). "
        "Must differ from --port. Default: 20010.",
    )
    parser.add_argument(
        "--no-multinode",
        action="store_true",
        help="Disable automatic multi-node mode even when config TP > local GPU count.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for inference requests (default: 0.0 = greedy). "
        "Set to e.g. 0.7 to enable top-k/top-p sampling for workload collection.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k value for sampling (default: -1 = no filtering). "
        "Set to e.g. 1000 to force top-k sampling path in SGLang.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) value for sampling (default: 1.0 = no filtering). "
        "Set to e.g. 0.9 to force top-p sampling path in SGLang.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=os.environ.get("CONDA_DEFAULT_ENV", "flashinfer_bench"),
        help="Conda environment to activate on peer nodes via SSH. "
        "Defaults to $CONDA_DEFAULT_ENV or 'flashinfer_bench'.",
    )
    parser.add_argument(
        "--restart-per-batch-size",
        action="store_true",
        help=(
            "Restart the SGLang server between batch sizes. "
            "Useful for workload collection: each server session gets its own "
            "FLASHINFER_DUMP_MAX_COUNT budget, preventing early batch sizes from "
            "exhausting the dump budget before later ones run."
        ),
    )
    args = parser.parse_args()

    # Resolve GPU type
    gpu_type = args.gpu if args.gpu else detect_gpu_type()
    log(f"GPU type:       {gpu_type}" + (" (auto-detected)" if not args.gpu else ""))

    model_config = get_model_config(args.model, gpu_type)
    num_prompts = args.batch_sizes[-1] * args.num_batches + 1

    log(f"Model type:     {args.model}")
    log(f"Model path:     {args.model_path}")
    log(f"Batch sizes:    {args.batch_sizes}")
    log(f"Total prompts:  {num_prompts} ({args.num_batches} batches per sweep)")

    prompts = load_prompts_from_sharegpt(num_prompts)

    server_args = []

    if args.disable_cuda_graph:
        server_args.append("--disable-cuda-graph")

    if args.disable_radix_cache:
        server_args.append("--disable-radix-cache")
        log("Added --disable-radix-cache for ragged prefill collection")

    if args.enable_deterministic_inference:
        server_args.append("--enable-deterministic-inference")
        log("Added --enable-deterministic-inference for paged prefill collection")

    server_args.extend(model_config["server_flags"])

    # --- TP size: always from model config (never overridden) ---
    # If config has no --tp-size, fall back to local GPU count.
    local_gpus = detect_tp_size() or 1

    config_tp: Optional[int] = None
    for i, arg in enumerate(server_args):
        if arg in ("--tp-size", "--tp") and i + 1 < len(server_args):
            try:
                config_tp = int(server_args[i + 1])
            except ValueError:
                pass
            break

    effective_tp = config_tp if config_tp is not None else local_gpus
    if config_tp is None:
        server_args = server_args + ["--tp-size", str(effective_tp)]
        log(f"TP size: {effective_tp} (no config TP, using local GPU count)")
    else:
        log(f"TP size: {effective_tp} (from config)")

    # --- Multi-node detection ---
    # Discover available peer nodes from SLURM allocation or --peer-node-addr.
    # Then compute how many nodes the config TP actually needs:
    #   needed_nodes = ceil(effective_tp / local_gpus)
    # Only use as many peers as needed, not all allocated nodes.
    # Example: 4+4 GPU allocation, TP=4 → needed_nodes=1 → single-node
    #          4+4 GPU allocation, TP=8 → needed_nodes=2 → multi-node
    needed_nodes = math.ceil(effective_tp / local_gpus)

    available_peers: List[Tuple[str, str]] = []
    if not args.no_multinode and needed_nodes > 1:
        if args.peer_node_addr:
            available_peers = [(addr, addr) for addr in args.peer_node_addr]
        else:
            available_peers = get_slurm_peer_nodes()

    peers = available_peers[: needed_nodes - 1]
    need_multinode = len(peers) > 0
    nnodes = 1 + len(peers)

    if needed_nodes > 1 and len(peers) < needed_nodes - 1:
        raise RuntimeError(
            f"TP={effective_tp} needs {needed_nodes} nodes ({local_gpus} GPUs each), "
            f"but only {1 + len(available_peers)} node(s) available. "
            f"Check SLURM allocation or pass --peer-node-addr."
        )

    if need_multinode:
        log(f"Multi-node: {nnodes} nodes × {local_gpus} GPUs, TP={effective_tp}")

    # --- Multi-node setup (compute once; reused across server sessions) ---
    head_ip: Optional[str] = None
    dist_addr: Optional[str] = None

    if need_multinode:
        head_ip = get_head_node_ip_for_peer(peers[0][1])
        dist_addr = f"{head_ip}:{args.dist_init_port}"
        log(f"dist-init-addr: {dist_addr}")
        log(f"Peer nodes:     {[p[0] for p in peers]}")

    log(f"Server args:    {server_args}")

    # Build server environment once: inherit os.environ, then override/add specific vars.
    # CPATH must come AFTER **os.environ so it takes precedence over any ambient value;
    # we prepend our nvidia includes to any existing CPATH so we don't shadow others.
    _existing_cpath = os.environ.get("CPATH", "")
    _server_cpath = (
        f"{_NVIDIA_CPATH}:{_existing_cpath}"
        if _existing_cpath and _NVIDIA_CPATH
        else (_NVIDIA_CPATH or _existing_cpath)
    )
    server_env: Dict[str, str] = {
        "SGLANG_RECORD_STEP_TIME": "1",
        "SGLANG_TEST_REQUEST_TIME_STATS": "1",
        **os.environ,
        **({"CPATH": _server_cpath} if _server_cpath else {}),
    }

    if args.restart_per_batch_size:
        # Per-batch-size isolation: each batch size gets its own server session and
        # therefore its own fresh FLASHINFER_DUMP_MAX_COUNT budget.
        log("restart-per-batch-size: ON — server will restart between batch sizes")
        for batch_size in args.batch_sizes:
            log(f"=== Starting server session for batch_size={batch_size} ===")
            process, peer_processes = _launch_server_session(
                model_path=args.model_path,
                base_url=args.base_url,
                server_args=server_args,
                server_env=server_env,
                need_multinode=need_multinode,
                peers=peers,
                nnodes=nnodes,
                head_ip=head_ip,
                dist_addr=dist_addr,
                dist_port=args.dist_init_port,
                conda_env=args.conda_env,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            )
            try:
                log(f"Running benchmark with batch size {batch_size}")
                run_benchmark(
                    args.base_url,
                    prompts[: batch_size * args.num_batches],
                    batch_size,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
            except Exception as e:
                log(f"Benchmark failed: {e}")
                raise
            finally:
                log(f"Shutting down server session for batch_size={batch_size}...")
                _shutdown_server(process, peer_processes)
    else:
        # Default: single server session for all batch sizes.
        process, peer_processes = _launch_server_session(
            model_path=args.model_path,
            base_url=args.base_url,
            server_args=server_args,
            server_env=server_env,
            need_multinode=need_multinode,
            peers=peers,
            nnodes=nnodes,
            head_ip=head_ip,
            dist_addr=dist_addr,
            dist_port=args.dist_init_port,
            conda_env=args.conda_env,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        try:
            for batch_size in args.batch_sizes:
                log(f"Running benchmark with batch size {batch_size}")
                run_benchmark(
                    args.base_url,
                    prompts[: batch_size * args.num_batches],
                    batch_size,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
        except Exception as e:
            log(f"Benchmark failed: {e}")
            raise
        finally:
            log("Shutting down server...")
            _shutdown_server(process, peer_processes)


if __name__ == "__main__":
    main()
