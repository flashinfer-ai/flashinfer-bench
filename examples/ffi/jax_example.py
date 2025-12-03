"""
JAX example: Load and run the distributed .so kernel using jax-tvm-ffi.

Requirements:
    pip install jax jax-tvm-ffi
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax_tvm_ffi
import tvm_ffi


def main():
    dist_dir = Path("distributed")
    so_path = dist_dir / "kernel.so"

    entry_symbol = None
    for line in (dist_dir / "kernel_metadata.txt").read_text().split("\n"):
        if line.startswith("Entry Symbol:"):
            entry_symbol = line.split(":", 1)[1].strip()
            break

    # Load and register kernel
    mod = tvm_ffi.load_module(str(so_path))
    kernel_fn = getattr(mod, entry_symbol)
    jax_tvm_ffi.register_ffi_target(entry_symbol, kernel_fn, platform="gpu")

    print(f"Loaded kernel: {entry_symbol}")

    # Prepare inputs: C = A @ B.T
    M, N, K = 1024, 4096, 4096
    jax_device = jax.devices("gpu")[0]
    A = jnp.array(
        jax.random.normal(jax.random.PRNGKey(0), (M, K)), dtype=jnp.float16, device=jax_device
    )
    B = jnp.array(
        jax.random.normal(jax.random.PRNGKey(1), (N, K)), dtype=jnp.float16, device=jax_device
    )

    result = jax.ffi.ffi_call(
        entry_symbol, jax.ShapeDtypeStruct((M, N), jnp.float16), vmap_method="broadcast_all"
    )(A, B)

    # Verify
    reference = jnp.matmul(A, B.T)
    max_diff = jnp.abs(result - reference).max()

    print(f"Max diff vs reference: {max_diff:.6f}")
    print(f"Correct: {max_diff < 1e-2}")


if __name__ == "__main__":
    main()
