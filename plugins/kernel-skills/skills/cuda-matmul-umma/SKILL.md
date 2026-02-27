---
name: cuda-matmul-umma
description: Write CUDA matmul kernels using UMMA (tcgen05.mma) instructions for Blackwell GPUs (sm100). Use when implementing matrix multiplication with 5th gen Tensor Cores, or when migrating from WGMMA to Blackwell architecture.
---

# UMMA (Unified Matrix Multiply-Accumulate) for Blackwell

UMMA (`tcgen05.mma`) is the 5th generation Tensor Core instruction on Blackwell (SM100), replacing Hopper's WGMMA. Key difference: single-thread launch with accumulator in Tensor Memory (TMEM).

## Key Differences from WGMMA

| Aspect | WGMMA (Hopper) | UMMA (Blackwell) |
|--------|----------------|------------------|
| Launch | 128 threads (warpgroup) | **1 thread** |
| Accumulator | Registers | **TMEM** |
| Max tile | 64×256×16 | **128×256×16** |
| A source | SMEM/Registers | SMEM/**TMEM** |
| Instruction | `wgmma.mma_async` | `tcgen05.mma` |

## Instruction Format

```
tcgen05.mma.cta_group.kind [d-tmem], a-desc, b-desc, idesc, enable-input-d;
tcgen05.mma.cta_group.kind [d-tmem], [a-tmem], b-desc, idesc, enable-input-d;

.kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4 }
.cta_group = { .cta_group::1, .cta_group::2 }
```

## Supported Shapes (FP16/BF16)

| Shape | M | N | K | Accum Size | Notes |
|-------|---|---|---|------------|-------|
| Small | 64 | 8-256 (×8) | 16 | 64×N | N multiple of 8 |
| Large | 128 | 16-256 (×16) | 16 | 128×N | **Best throughput** |

## Basic UMMA Usage

```cuda
// UMMA is single-threaded - only one thread issues the instruction
__device__ void umma_m128n256k16(
    uint32_t tmem_d_addr,  // TMEM address for accumulator D
    uint64_t desc_a,       // SMEM descriptor for A (or TMEM address)
    uint64_t desc_b,       // SMEM descriptor for B
    uint32_t idesc,        // Instruction descriptor
    bool accumulate
) {
    int scale_d = accumulate ? 1 : 0;

    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 "
        "[%0], %1, %2, %3, %4;\n"
        :
        : "r"(tmem_d_addr), "l"(desc_a), "l"(desc_b),
          "r"(idesc), "r"(scale_d)
        : "memory"
    );
}
```

## Instruction Descriptor

32-bit metadata encoding data types, sparsity, and transpose:

```cuda
__device__ uint32_t make_umma_idesc(
    bool transpose_a = false,
    bool transpose_b = false,
    bool negate_a = false,
    bool negate_b = false
) {
    uint32_t idesc = 0;

    // Bits for transpose (check PTX spec for exact positions)
    if (transpose_a) idesc |= (1 << 0);
    if (transpose_b) idesc |= (1 << 1);
    if (negate_a)    idesc |= (1 << 2);
    if (negate_b)    idesc |= (1 << 3);

    // Data type encoding for FP16 (check spec)
    // ...

    return idesc;
}
```

## Complete UMMA Kernel Pattern

```cuda
__global__ void matmul_umma(
    const CUtensorMap* __restrict__ tma_a,
    const CUtensorMap* __restrict__ tma_b,
    CUtensorMap* __restrict__ tma_c,
    int M, int N, int K
) {
    extern __shared__ char smem[];

    // TMEM allocation (one warp does this)
    __shared__ uint32_t tmem_base_ptr;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    bool elect_one_warp = (warp_id == 0);
    bool elect_one_thr = (lane_id == 0);

    // Allocate TMEM (512 columns = full capacity)
    if (elect_one_warp) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], 512;\n"
            :
            : "l"((uint64_t)&tmem_base_ptr)
        );
    }
    __syncthreads();

    // Initialize mbarrier
    __shared__ uint64_t mma_barrier;
    if (elect_one_warp && elect_one_thr) {
        mbarrier_init(&mma_barrier, 1);  // Single CTA
    }
    __syncthreads();

    int mma_phase = 0;
    uint32_t idesc = make_umma_idesc();

    // First MMA: overwrite accumulator
    bool first_k = true;

    for (int k = 0; k < K / TILE_K; k++) {
        // TMA loads (same as Hopper)
        // ...

        // Only one warp issues UMMAs
        if (elect_one_warp) {
            for (int k_inner = 0; k_inner < TILE_K / 16; k_inner++) {
                uint64_t desc_a = make_smem_desc(smem_a, k_inner);
                uint64_t desc_b = make_smem_desc(smem_b, k_inner);

                umma_m128n256k16(
                    tmem_base_ptr,
                    desc_a, desc_b,
                    idesc,
                    !first_k  // accumulate after first
                );
                first_k = false;
            }

            // Signal completion via mbarrier
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
                :
                : "l"((uint64_t)&mma_barrier)
            );
        }

        // All warps wait for MMA completion before reusing SMEM
        mbarrier_wait(&mma_barrier, mma_phase);
        mma_phase ^= 1;
    }

    // Copy accumulator from TMEM to registers for epilogue
    float accum[...];
    // Use tcgen05.ld (see cuda-matmul-tmem-load)

    // Store results
    // ...

    // Deallocate TMEM
    if (elect_one_warp) {
        asm volatile("tcgen05.relinquish_alloc_permit;\n" ::);
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;\n"
            :
            : "r"(tmem_base_ptr)
        );
    }
}
```

## SMEM Descriptor (Similar to WGMMA)

```cuda
__device__ uint64_t make_smem_desc_b128(void* smem_ptr) {
    uint64_t desc = 0;
    uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);

    // Address bits
    desc |= (addr >> 4) & 0x3FFF;

    // B128 swizzle (same as Hopper)
    desc |= (uint64_t)3 << 62;

    return desc;
}
```

## Synchronization

UMMA completion tracked via `tcgen05.commit`:

```cuda
// After issuing UMMA(s)
asm volatile(
    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
    :
    : "l"(mbar_addr)
);

// Wait for completion
mbarrier_wait(&mbar, phase);
```

## Migration from WGMMA

1. **Accumulator**: Move from registers to TMEM
2. **Launch**: Change from warpgroup to single thread
3. **Allocation**: Add TMEM alloc/dealloc
4. **Epilogue**: Add `tcgen05.ld` to copy results from TMEM

## Key Points

1. **Single thread launch**: Only one thread issues `tcgen05.mma`
2. **TMEM required**: Accumulator must be in Tensor Memory
3. **Larger tiles**: 128×256 vs 64×256 for better efficiency
4. **mbarrier commit**: Use `tcgen05.commit` for completion signaling
5. **Explicit dealloc**: Must free TMEM before kernel exit

## Related Skills
- `cuda-matmul-tmem`: Tensor Memory allocation and management
- `cuda-matmul-tmem-load`: Moving data out of TMEM
- `cuda-matmul-cta-pair`: 2-SM UMMA with CTA pairs
- `cuda-matmul-wgmma`: Hopper equivalent (for comparison)
