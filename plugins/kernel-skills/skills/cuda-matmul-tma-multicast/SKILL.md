---
name: cuda-matmul-tma-multicast
description: Use TMA multicast to broadcast matrix tiles across SM clusters in CUDA matmul on Hopper+ GPUs. Use when sharing B matrix tiles across CTAs, or when reducing global memory bandwidth for GEMM on sm90+.
---

# TMA Multicast for Matmul

TMA multicast broadcasts data from global memory to multiple CTAs' shared memory in a single operation, reducing bandwidth for shared tiles.

## Multicast Concept

In a 2×1 cluster for matmul:
- Both CTAs compute different M tiles but same N tile
- B matrix tile (K×N) is identical for both CTAs
- Multicast: Load B once, deliver to both CTAs simultaneously

```
Without multicast:      With multicast:
CTA0: Load B -> SMEM0   CTA0: Load B -> SMEM0 + SMEM1
CTA1: Load B -> SMEM1   CTA1: (no load needed)
Bandwidth: 2x           Bandwidth: 1x
```

## TMA Descriptor for Multicast

Host-side setup (same as regular TMA):

```cuda
CUtensorMap create_tma_multicast_desc(
    void* gmem_ptr,
    int K, int N,
    int tile_k, int tile_n
) {
    CUtensorMap tensor_map{};

    uint64_t size[2] = {(uint64_t)N, (uint64_t)K};
    uint64_t stride[1] = {(uint64_t)N * sizeof(__nv_bfloat16)};
    uint32_t box_size[2] = {(uint32_t)tile_n, (uint32_t)tile_k};
    uint32_t elem_stride[2] = {1, 1};

    cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        gmem_ptr,
        size,
        stride,
        box_size,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_OOBFILL_ZERO
    );

    return tensor_map;
}
```

## Multicast Load Operation

```cuda
__device__ void tma_load_multicast_2d(
    void* smem_ptr,
    const CUtensorMap* tensor_map,
    int coord_n, int coord_k,
    uint64_t* mbar,
    uint16_t multicast_mask  // Bitmask of destination CTAs
) {
    uint64_t smem_addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t mbar_addr = (uint64_t)__cvta_generic_to_shared(mbar);

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3}], [%4], %5;\n"
        :
        : "l"(smem_addr), "l"(tensor_map),
          "r"(coord_n), "r"(coord_k),
          "l"(mbar_addr), "h"(multicast_mask)
        : "memory"
    );
}
```

## Multicast Mask

The mask specifies which CTAs receive the data:

```cuda
// For 2x1 cluster (2 CTAs)
uint16_t mask_both = 0b11;      // CTA 0 and 1
uint16_t mask_cta0 = 0b01;      // CTA 0 only
uint16_t mask_cta1 = 0b10;      // CTA 1 only

// For 2x2 cluster (4 CTAs)
uint16_t mask_all = 0b1111;     // All 4 CTAs
uint16_t mask_row0 = 0b0011;    // CTA 0,1 (first row)
uint16_t mask_col0 = 0b0101;    // CTA 0,2 (first column)
```

## Complete Multicast Pattern

```cuda
constexpr int CLUSTER_M = 2;
constexpr int CLUSTER_N = 1;

__global__ void matmul_multicast(
    const CUtensorMap* tma_a,
    const CUtensorMap* tma_b,  // B will be multicast
    float* C,
    int M, int N, int K
) {
    extern __shared__ char smem[];
    auto* shared = reinterpret_cast<SharedStorage*>(smem);

    int cta_rank = cta_rank_in_cluster();
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Only CTA 0 loads B with multicast
    bool is_b_loader = (cta_rank == 0);

    // Multicast mask: both CTAs in 2x1 cluster
    uint16_t multicast_mask = (1 << CLUSTER_M) - 1;  // 0b11

    for (int k = 0; k < K / TILE_K; k++) {
        int stage = k % NUM_STAGES;

        if (warp_id == 0 && lane_id == 0) {
            // Each CTA loads its own A tile
            size_t a_bytes = TILE_M * TILE_K * sizeof(__nv_bfloat16);
            mbarrier_arrive_expect_tx(&shared->mbar[stage], a_bytes);
            tma_load_2d(shared->smem_a[stage], tma_a,
                k * TILE_K, (m_tile * CLUSTER_M + cta_rank) * TILE_M,
                &shared->mbar[stage]);

            // Only CTA 0 loads B with multicast
            if (is_b_loader) {
                size_t b_bytes = TILE_K * TILE_N * sizeof(__nv_bfloat16);
                // Expect bytes on all destination CTAs' barriers
                mbarrier_arrive_expect_tx(&shared->mbar[stage], b_bytes);

                tma_load_multicast_2d(
                    shared->smem_b[stage], tma_b,
                    n_tile * TILE_N, k * TILE_K,
                    &shared->mbar[stage],
                    multicast_mask
                );
            }
        }

        // Wait for both A and B loads
        mbarrier_wait(&shared->mbar[stage], phase);

        // WGMMA computation
        // ...
    }
}
```

## Barrier Handling for Multicast

Critical: All destination CTAs must set expected bytes:

```cuda
// CTA 0 (loader)
if (cta_rank == 0 && lane_id == 0) {
    mbarrier_arrive_expect_tx(&mbar, b_tile_bytes);
    tma_load_multicast(..., multicast_mask);
}

// CTA 1 (receiver) - MUST also expect bytes!
if (cta_rank == 1 && lane_id == 0) {
    mbarrier_arrive_expect_tx(&mbar, b_tile_bytes);
    // No TMA load, but barrier expects incoming data
}
```

## Bandwidth Savings

| Cluster | Multicast | Bandwidth Reduction |
|---------|-----------|---------------------|
| 1×1 | None | 0% |
| 2×1 | B tile | **50% for B** |
| 1×2 | A tile | 50% for A |
| 2×2 | Both | 50% for A and B |

For matmul with large N: B tile dominates, 2×1 cluster most effective.

## SMEM Layout Considerations

Multicast delivers to same SMEM offset in all CTAs:

```cuda
// Both CTA 0 and CTA 1 receive B at same local offset
__shared__ __nv_bfloat16 smem_b[TILE_K * TILE_N];

// Multicast writes to smem_b in both CTAs
tma_load_multicast_2d(smem_b, tma_b, ..., mask_both);

// After sync, both CTAs can read from their local smem_b
```

## Key Points

1. **Single issuer**: Only one CTA issues multicast, others just wait
2. **All expect bytes**: Every destination CTA must call arrive_expect_tx
3. **Same SMEM offset**: Data lands at identical offset in all CTAs
4. **Cluster required**: Must launch with cluster configuration
5. **Mask ordering**: Bit 0 = CTA rank 0, Bit 1 = CTA rank 1, etc.

## Related Skills
- `cuda-matmul-cluster`: Cluster configuration and coordination
- `cuda-matmul-tma`: Basic TMA operations
- `cuda-matmul-barrier`: Multicast barrier synchronization
