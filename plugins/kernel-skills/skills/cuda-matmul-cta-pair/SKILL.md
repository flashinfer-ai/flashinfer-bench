---
name: cuda-matmul-cta-pair
description: Use CTA pairs for 2-SM UMMA operations in CUDA matmul on Blackwell GPUs (sm100). Use when implementing cooperative matrix multiplication across two SMs, or when maximizing Tensor Core utilization with paired CTAs.
---

# CTA Pairs for 2-SM UMMA on Blackwell

Blackwell introduces CTA pairs: two adjacent CTAs within a cluster that cooperate on a single UMMA operation across two SMs, doubling compute throughput per operation.

## CTA Pair Concept

```
Cluster (e.g., 2×1):
┌─────────────┬─────────────┐
│   CTA 0     │   CTA 1     │
│   (SM 0)    │   (SM 1)    │
│             │             │
│  UMMA half  │  UMMA half  │
│  of tile    │  of tile    │
└─────────────┴─────────────┘
        ↓
   Combined UMMA output
   (2× throughput per instruction)
```

## Key Differences from Single-CTA

| Aspect | 1-CTA UMMA | 2-CTA (CTA Pair) UMMA |
|--------|------------|----------------------|
| Instruction | `tcgen05.mma.cta_group::1` | `tcgen05.mma.cta_group::2` |
| SMs involved | 1 | 2 |
| Launch thread | 1 in CTA | **1 in one CTA only** |
| Output | Full tile in TMEM | Split across both CTAs' TMEM |
| Throughput | 1× | **2×** |

## CTA Pair UMMA Instruction

```cuda
// Only CTA 0 (peer rank 0) issues the instruction
// CTA 1 participates implicitly

__device__ void umma_2sm_m128n256k16(
    uint32_t tmem_d_addr,  // TMEM in this CTA
    uint64_t desc_a,       // A descriptor
    uint64_t desc_b,       // B descriptor
    uint32_t idesc,
    bool accumulate
) {
    int scale_d = accumulate ? 1 : 0;

    asm volatile(
        "tcgen05.mma.cta_group::2.kind::f16 "  // Note: cta_group::2
        "[%0], %1, %2, %3, %4;\n"
        :
        : "r"(tmem_d_addr), "l"(desc_a), "l"(desc_b),
          "r"(idesc), "r"(scale_d)
        : "memory"
    );
}
```

## Peer CTA Identification

```cuda
__device__ int get_peer_cta_rank() {
    // Position within CTA pair (0 or 1)
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    return rank % 2;  // For 2-CTA pairs
}

__device__ bool is_umma_launcher() {
    // Only peer rank 0 launches 2-SM UMMA
    return get_peer_cta_rank() == 0;
}
```

## Output Distribution

The 2-SM UMMA splits output across both CTAs:

```
For 128×256 output tile:
- CTA 0 gets rows 0-63 in its TMEM
- CTA 1 gets rows 64-127 in its TMEM

Each CTA processes its portion for epilogue.
```

## Complete 2-SM UMMA Pattern

```cuda
__global__ void matmul_cta_pair(
    const CUtensorMap* __restrict__ tma_a,
    const CUtensorMap* __restrict__ tma_b,
    float* C,
    int M, int N, int K
) {
    extern __shared__ char smem[];

    int peer_rank = get_peer_cta_rank();
    bool is_launcher = (peer_rank == 0);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    bool elect_warp = (warp_id == 0);
    bool elect_thr = (lane_id == 0);

    // TMEM allocation (each CTA allocates its own)
    __shared__ uint32_t tmem_base;
    if (elect_warp) {
        // cta_group::2 for paired allocation
        asm volatile(
            "tcgen05.alloc.cta_group::2.sync.aligned.b32 [%0], 256;\n"
            :
            : "l"((uint64_t)__cvta_generic_to_shared(&tmem_base))
        );
    }
    __syncthreads();

    // Barrier for 2-CTA synchronization
    __shared__ uint64_t mma_barrier;
    if (elect_warp && elect_thr) {
        // Initialize for 2 CTAs
        mbarrier_init(&mma_barrier, 2);
    }
    __syncthreads();

    int mma_phase = 0;
    bool first_k = true;

    // Tile coordinates (both CTAs work on same tile)
    int m_tile = blockIdx.x / 2;  // Pair shares M tile
    int n_tile = blockIdx.y;

    for (int k = 0; k < K / TILE_K; k++) {
        // TMA loads - can use multicast for B
        // Each CTA loads its portion of A
        if (elect_warp && elect_thr) {
            // Load A (each CTA loads different M portion)
            int m_offset = m_tile * 128 + peer_rank * 64;
            tma_load_2d(smem_a, tma_a, k * TILE_K, m_offset, &load_bar);

            // Load B (shared via multicast or both load)
            if (peer_rank == 0) {
                tma_load_multicast_2d(smem_b, tma_b, n_tile * 256, k * TILE_K,
                                       &load_bar, 0b11);
            }
        }

        // Wait for loads
        mbarrier_wait(&load_bar, phase);

        // UMMA - only CTA 0 launches, but both participate
        if (elect_warp && is_launcher) {
            for (int k_inner = 0; k_inner < TILE_K / 16; k_inner++) {
                uint64_t desc_a = make_smem_desc(smem_a, k_inner);
                uint64_t desc_b = make_smem_desc(smem_b, k_inner);

                umma_2sm_m128n256k16(
                    tmem_base,
                    desc_a, desc_b,
                    idesc,
                    !first_k
                );
                first_k = false;
            }

            // Commit with 2-CTA barrier
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];\n"
                :
                : "l"((uint64_t)__cvta_generic_to_shared(&mma_barrier))
            );
        }

        // Both CTAs wait
        mbarrier_wait(&mma_barrier, mma_phase);
        mma_phase ^= 1;
    }

    // Epilogue: Each CTA loads its portion from TMEM
    // CTA 0: rows 0-63, CTA 1: rows 64-127
    float accum[64 * 256 / 128];  // Per-thread portion

    // Load from TMEM
    load_tmem_portion(tmem_base, accum, peer_rank);

    // Store to global (different rows per CTA)
    int row_offset = m_tile * 128 + peer_rank * 64;
    store_to_global(accum, C, row_offset, n_tile * 256, M, N);

    // Deallocate TMEM
    if (elect_warp) {
        asm volatile("tcgen05.relinquish_alloc_permit;\n" ::);
        asm volatile(
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 256;\n"
            :
            : "r"(tmem_base)
        );
    }
}
```

## Launch Configuration

```cuda
// Must launch with cluster that enables CTA pairs
cudaLaunchConfig_t config = {};
config.blockDim = dim3(BLOCK_SIZE);
config.gridDim = dim3(M_tiles * 2, N_tiles);  // 2 CTAs per M tile

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {2, 1, 1};  // 2×1 cluster for CTA pairs

config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, matmul_cta_pair, ...);
```

## Synchronization for CTA Pairs

```cuda
// 2-CTA barrier operations
__device__ void mbarrier_init_2cta(uint64_t* mbar, int threads_per_cta) {
    mbarrier_init(mbar, threads_per_cta * 2);
}

// Arrive on partner CTA's barrier
__device__ void arrive_remote_cta(uint64_t* mbar, int partner_rank) {
    uint64_t local_addr = (uint64_t)__cvta_generic_to_shared(mbar);
    uint64_t remote_addr;

    asm volatile(
        "mapa.shared::cluster.u64 %0, %1, %2;\n"
        : "=l"(remote_addr)
        : "l"(local_addr), "r"(partner_rank)
    );

    asm volatile(
        "mbarrier.arrive.shared::cluster.b64 _, [%0];\n"
        :
        : "l"(remote_addr)
        : "memory"
    );
}
```

## Benefits of CTA Pairs

| Benefit | Description |
|---------|-------------|
| **2× Throughput** | Single instruction uses 2 SMs' Tensor Cores |
| **Shared B tile** | Multicast B to both CTAs |
| **Reduced overhead** | One launch, two SMs |
| **Natural output split** | Each CTA handles half of output |

## Key Points

1. **Single launcher**: Only peer rank 0 issues `tcgen05.mma.cta_group::2`
2. **Both allocate TMEM**: Each CTA needs its own TMEM for output half
3. **Cluster required**: Must launch with appropriate cluster dimensions
4. **Output split**: 128×256 tile splits to 64×256 per CTA
5. **Coordinated barriers**: Use 2-CTA barriers for synchronization

## Related Skills
- `cuda-matmul-umma`: Single-CTA UMMA
- `cuda-matmul-tmem`: TMEM allocation for CTA pairs
- `cuda-matmul-cluster`: Cluster configuration
- `cuda-matmul-tma-multicast`: Sharing B across CTA pair
