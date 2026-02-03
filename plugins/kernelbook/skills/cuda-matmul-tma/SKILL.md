---
name: cuda-matmul-tma
description: Use TMA (Tensor Memory Accelerator) for efficient global-to-shared memory transfers in CUDA matmul on Hopper+ GPUs. Use when loading matrix tiles asynchronously, or when optimizing memory bandwidth for GEMM kernels on sm90+.
---

# TMA (Tensor Memory Accelerator)

TMA provides hardware-accelerated tensor data movement on SM90+. It handles addressing, bounds checking, and format conversion automatically.

## TMA Descriptor Setup (Host Side)

```cuda
#include <cuda.h>

CUtensorMap create_tma_descriptor(
    void* gmem_ptr,           // Global memory pointer
    int M, int N,             // Tensor dimensions
    int tile_m, int tile_n,   // Tile dimensions
    cudaDataType dtype = CUDA_R_16BF
) {
    CUtensorMap tensor_map{};

    // Tensor dimensions (up to 5D supported)
    uint64_t size[2] = {(uint64_t)N, (uint64_t)M};  // Column-major
    uint64_t stride[1] = {(uint64_t)N * sizeof(__nv_bfloat16)};  // Stride in bytes

    // Tile box dimensions
    uint32_t box_size[2] = {(uint32_t)tile_n, (uint32_t)tile_m};
    uint32_t elem_stride[2] = {1, 1};

    cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,                              // Rank
        gmem_ptr,
        size,
        stride,
        box_size,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,     // B128 swizzle for WGMMA
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_OOBFILL_ZERO      // Zero-fill out-of-bounds
    );

    return tensor_map;
}
```

## TMA Load (Device Side)

### Single CTA Load

```cuda
__device__ void tma_load_2d(
    void* smem_ptr,
    const CUtensorMap* tensor_map,
    int coord_n, int coord_m,     // Tile coordinates (not byte offsets!)
    uint64_t* mbar                 // mbarrier for completion tracking
) {
    uint64_t smem_addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t mbar_addr = (uint64_t)__cvta_generic_to_shared(mbar);

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :
        : "l"(smem_addr), "l"(tensor_map), "r"(coord_n), "r"(coord_m), "l"(mbar_addr)
        : "memory"
    );
}
```

### Coordinates vs Offsets
- TMA uses **element coordinates**, not byte offsets
- For loading tile at row `m`, col `n`: pass `{n, m}` (column-major order)

## TMA Store

```cuda
__device__ void tma_store_2d(
    const CUtensorMap* tensor_map,
    void* smem_ptr,
    int coord_n, int coord_m
) {
    uint64_t smem_addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);

    // Fence before store
    asm volatile("fence.proxy.async.shared::cta;\n" ::);

    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];\n"
        :
        : "l"(tensor_map), "r"(coord_n), "r"(coord_m), "l"(smem_addr)
        : "memory"
    );

    // Commit and wait
    asm volatile("cp.async.bulk.commit_group;\n" ::);
    asm volatile("cp.async.bulk.wait_group 0;\n" ::);
}
```

## Swizzle Modes

| Mode | Bytes | Use Case |
|------|-------|----------|
| `SWIZZLE_NONE` | - | Debug only |
| `SWIZZLE_32B` | 32 | Narrow tiles |
| `SWIZZLE_64B` | 64 | Medium tiles |
| `SWIZZLE_128B` | 128 | **WGMMA (recommended)** |

## Expected Bytes Calculation

For mbarrier transaction tracking:

```cuda
// Tile size in bytes for arrive-on-expect
size_t expected_bytes = tile_m * tile_n * sizeof(__nv_bfloat16);

// Set expected transaction count before TMA load
mbarrier_arrive_expect_tx(mbar, expected_bytes);
```

## Common Patterns

### Loading A and B Tiles

```cuda
// A tile: M×K, B tile: K×N
constexpr int TILE_M = 128, TILE_N = 256, TILE_K = 64;

__shared__ __nv_bfloat16 smem_a[TILE_M * TILE_K];
__shared__ __nv_bfloat16 smem_b[TILE_K * TILE_N];

// Load A tile at (m_idx, k_idx)
tma_load_2d(smem_a, &tma_a, k_idx * TILE_K, m_idx * TILE_M, &mbar);

// Load B tile at (k_idx, n_idx)
tma_load_2d(smem_b, &tma_b, n_idx * TILE_N, k_idx * TILE_K, &mbar);
```

## Key Points

1. **Single thread**: Only one thread issues TMA (typically thread 0)
2. **Async completion**: Use mbarrier to track completion
3. **Alignment**: SMEM pointer must match swizzle alignment (128B for B128)
4. **Bounds handling**: TMA auto-fills zeros for out-of-bounds accesses

## Related Skills
- `cuda-matmul-tma-multicast`: Multicast loads across cluster
- `cuda-matmul-barrier`: mbarrier synchronization
- `cuda-matmul-swizzle`: SMEM layout matching TMA swizzle
