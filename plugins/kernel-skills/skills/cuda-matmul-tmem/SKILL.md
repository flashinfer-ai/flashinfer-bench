---
name: cuda-matmul-tmem
description: Manage Tensor Memory (TMEM) for CUDA matmul on Blackwell GPUs (sm100). Use when allocating accumulator storage for UMMA, or when working with 5th gen Tensor Core operations that require dedicated tensor memory.
---

# Tensor Memory (TMEM) for Blackwell

TMEM is a dedicated 256KB on-chip memory per SM for Tensor Core operations on Blackwell. It replaces registers for UMMA accumulators, reducing register pressure.

## TMEM Specifications

| Property | Value |
|----------|-------|
| Size per SM | 256 KB |
| Organization | 512 columns × 128 lanes |
| Cell size | 32 bits |
| Address format | 32-bit: [31:16]=lane, [15:0]=column |
| Allocation unit | Columns (power of 2, min 32) |

## TMEM Layout

```
        Column 0    Column 1    ...    Column 511
Lane 0  [32 bits]   [32 bits]   ...    [32 bits]
Lane 1  [32 bits]   [32 bits]   ...    [32 bits]
...
Lane 127[32 bits]   [32 bits]   ...    [32 bits]

Address = (lane_id << 16) | column_id
```

## TMEM Allocation

```cuda
// Allocate TMEM - returns base address to SMEM location
__device__ void tmem_alloc(uint32_t* smem_addr_ptr, int num_columns) {
    // num_columns must be power of 2, >= 32
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;\n"
        :
        : "l"((uint64_t)smem_addr_ptr), "r"(num_columns)
    );
}

// Full capacity allocation (512 columns = 256KB)
__device__ void tmem_alloc_full(uint32_t* smem_addr_ptr) {
    tmem_alloc(smem_addr_ptr, 512);
}
```

## TMEM Deallocation

```cuda
// Must be called by same warp that allocated
__device__ void tmem_free(uint32_t tmem_base, int num_columns) {
    // Optional: signal no more allocations from this CTA
    asm volatile("tcgen05.relinquish_alloc_permit;\n" ::);

    // Free the allocation
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
        :
        : "r"(tmem_base), "r"(num_columns)
    );
}
```

## TMEM Address Calculation

```cuda
// TMEM uses 2D addressing: lane (row) and column
__device__ uint32_t tmem_addr(uint32_t base, int lane, int col) {
    // Lane in upper 16 bits, column in lower 16 bits
    return base + (lane << 16) + col;
}

// For UMMA accumulator offset
__device__ uint32_t tmem_accum_offset(uint32_t base, int m_offset, int n_offset) {
    // Layout depends on UMMA tile size
    // For 128x256 tile: M maps to lanes, N maps to columns
    return base + (m_offset << 16) + n_offset;
}
```

## Allocation Patterns

### Single UMMA Tile (128×256 FP32)

```cuda
// 128×256 FP32 accumulator = 128 lanes × 256 columns × 4 bytes
// = 128 KB (half of TMEM)
constexpr int ACCUM_COLS = 256;

__shared__ uint32_t tmem_base;

if (warp_id == 0) {
    tmem_alloc(&tmem_base, ACCUM_COLS);
}
__syncthreads();

// Use tmem_base as accumulator address for UMMA
```

### Double Buffering

```cuda
// Two accumulator buffers for pipelining
constexpr int ACCUM_COLS = 256;
constexpr int NUM_BUFFERS = 2;

__shared__ uint32_t tmem_base[NUM_BUFFERS];

if (warp_id == 0) {
    tmem_alloc(&tmem_base[0], ACCUM_COLS * NUM_BUFFERS);
    tmem_base[1] = tmem_base[0] + ACCUM_COLS;
}
__syncthreads();
```

## TMEM Capacity Planning

| UMMA Tile | FP32 Accum Size | Columns | % of TMEM |
|-----------|-----------------|---------|-----------|
| 64×64 | 16 KB | 64 | 12.5% |
| 64×128 | 32 KB | 128 | 25% |
| 64×256 | 64 KB | 256 | 50% |
| 128×128 | 64 KB | 256 | 50% |
| 128×256 | 128 KB | 256* | 50% |

*128×256 uses 128 lanes × 256 cols, fitting in 256 columns with all lanes active.

## Complete Allocation Pattern

```cuda
__global__ void kernel_with_tmem() {
    extern __shared__ char smem[];

    // TMEM base address storage
    __shared__ uint32_t tmem_base_ptr;

    int warp_id = threadIdx.x / 32;
    bool elect_warp = (warp_id == 0);

    // === Allocate TMEM (one warp) ===
    if (elect_warp) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], 512;\n"
            :
            : "l"((uint64_t)__cvta_generic_to_shared(&tmem_base_ptr))
        );
    }
    __syncthreads();

    // tmem_base_ptr now contains the TMEM base address
    uint32_t tmem_accum = tmem_base_ptr;

    // === Use TMEM for UMMA ===
    // ... UMMA operations with tmem_accum as accumulator ...

    // === Copy results from TMEM to registers ===
    // (see cuda-matmul-tmem-load)

    // === Deallocate TMEM (same warp) ===
    if (elect_warp) {
        asm volatile("tcgen05.relinquish_alloc_permit;\n" ::);
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;\n"
            :
            : "r"(tmem_base_ptr)
        );
    }
}
```

## Debugging TMEM Access

Compile with `--g-tensor-memory-access-check` to catch:
- Uninitialized TMEM reads
- Out-of-bounds access
- Use after dealloc

```bash
nvcc --g-tensor-memory-access-check -arch=sm_100 kernel.cu
```

## Key Points

1. **Warp allocation**: Same warp must alloc and dealloc
2. **SMEM address storage**: `tcgen05.alloc` writes to SMEM location
3. **Power-of-2 columns**: Allocation size must be 32, 64, 128, 256, or 512
4. **No direct compute**: Must copy to registers for post-processing
5. **Lane restrictions**: Each warp accesses only 32 of 128 lanes

## Related Skills
- `cuda-matmul-umma`: Using TMEM with UMMA operations
- `cuda-matmul-tmem-load`: Data movement from TMEM
- `cuda-matmul-cta-pair`: TMEM sharing across CTA pairs
