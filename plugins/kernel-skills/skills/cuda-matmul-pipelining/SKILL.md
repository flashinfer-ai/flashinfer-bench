---
name: cuda-matmul-pipelining
description: Implement software pipelining for CUDA matmul to overlap memory transfers with computation on Hopper+ GPUs. Use when designing multi-stage GEMM mainloops, or when hiding memory latency with async operations on sm90+.
---

# Software Pipelining for Matmul

Pipelining overlaps TMA loads with WGMMA computation across multiple stages, hiding memory latency.

## Pipeline Concept

```
Stage 0: [Load A0,B0] [Compute --] [Load A1,B1] [Compute C0] [Load A2,B2] [Compute C1] ...
Stage 1:             [Load A1,B1] [Compute --] [Load A2,B2] [Compute C0] ...
                                               ^^^^^^^^^^^^ ^^^^^^^^^^^
                                               Overlapped operations
```

## Multi-Stage Pipeline Structure

```cuda
constexpr int NUM_STAGES = 4;  // Typical: 2-4 stages

// Circular buffer for tiles
__shared__ __nv_bfloat16 smem_a[NUM_STAGES][TILE_M * TILE_K];
__shared__ __nv_bfloat16 smem_b[NUM_STAGES][TILE_K * TILE_N];

// One mbarrier per stage
__shared__ uint64_t mbar_load[NUM_STAGES];
__shared__ uint64_t mbar_compute[NUM_STAGES];
```

## Pipeline State Machine

```cuda
struct PipelineState {
    int producer_stage;   // Next stage to load into
    int consumer_stage;   // Next stage to compute from
    int k_tile;          // Current K iteration
    int num_k_tiles;     // Total K tiles

    __device__ bool has_work() { return k_tile < num_k_tiles + NUM_STAGES - 1; }
    __device__ bool can_produce() { return k_tile < num_k_tiles; }
    __device__ bool can_consume() { return k_tile >= NUM_STAGES - 1; }

    __device__ void advance() {
        producer_stage = (producer_stage + 1) % NUM_STAGES;
        consumer_stage = (consumer_stage + 1) % NUM_STAGES;
        k_tile++;
    }
};
```

## Mainloop with Pipeline

```cuda
__device__ void gemm_mainloop(
    const CUtensorMap* tma_a,
    const CUtensorMap* tma_b,
    int m_tile, int n_tile, int num_k_tiles,
    float* accum
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    bool is_producer = (warp_id == 0);  // Warp 0 does TMA loads

    PipelineState pipe = {0, 0, 0, num_k_tiles};

    // Prologue: Fill pipeline stages
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (is_producer && lane_id == 0 && s < num_k_tiles) {
            // Set expected bytes
            mbarrier_arrive_expect_tx(&mbar_load[s],
                TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2);

            // Issue TMA loads
            tma_load_2d(smem_a[s], tma_a, s * TILE_K, m_tile * TILE_M, &mbar_load[s]);
            tma_load_2d(smem_b[s], tma_b, n_tile * TILE_N, s * TILE_K, &mbar_load[s]);
        }
        pipe.k_tile++;
        pipe.producer_stage = (pipe.producer_stage + 1) % NUM_STAGES;
    }
    __syncthreads();

    // Mainloop: Steady state
    while (pipe.has_work()) {
        int prod_stage = pipe.producer_stage;
        int cons_stage = pipe.consumer_stage;

        // Producer: Load next tile
        if (is_producer && lane_id == 0 && pipe.can_produce()) {
            mbarrier_arrive_expect_tx(&mbar_load[prod_stage],
                TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2);

            int k = pipe.k_tile - (NUM_STAGES - 1);
            tma_load_2d(smem_a[prod_stage], tma_a, k * TILE_K, m_tile * TILE_M, &mbar_load[prod_stage]);
            tma_load_2d(smem_b[prod_stage], tma_b, n_tile * TILE_N, k * TILE_K, &mbar_load[prod_stage]);
        }

        // Consumer: Wait for data and compute
        if (pipe.can_consume()) {
            // Wait for load to complete
            mbarrier_wait(&mbar_load[cons_stage]);

            // Build descriptors and compute WGMMA
            uint64_t desc_a = make_gmma_desc(smem_a[cons_stage]);
            uint64_t desc_b = make_gmma_desc(smem_b[cons_stage]);

            // Issue WGMMA (simplified - actual has multiple K iterations)
            wgmma_m64n256k16(desc_a, desc_b, accum, /*accumulate=*/true);

            // Commit WGMMA
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::);

            // Signal compute done (producer can reuse buffer)
            mbarrier_arrive(&mbar_compute[cons_stage]);
        }

        pipe.advance();
    }

    // Epilogue: Wait for all WGMMA to complete
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::);
}
```

## Barrier Synchronization Pattern

```cuda
// Producer side (TMA warp)
mbarrier_arrive_expect_tx(&mbar, expected_bytes);  // Declare expected bytes
tma_load(..., &mbar);                              // TMA adds to transaction count
// TMA hardware signals completion when bytes arrive

// Consumer side (Math warps)
mbarrier_wait(&mbar);  // Block until TMA completes
// Safe to read SMEM now
```

## Stage Count Selection

| Stages | SMEM Usage | Latency Hiding | Use Case |
|--------|-----------|----------------|----------|
| 2 | Low | Minimal | Register-limited |
| 3 | Medium | Good | Balanced |
| 4 | High | Best | **Recommended** |
| 5+ | Very High | Diminishing | Large tiles |

## Memory Layout for Pipeline

```cuda
// Efficient layout: Stages interleaved for locality
__shared__ struct {
    __nv_bfloat16 a[TILE_M * TILE_K];
    __nv_bfloat16 b[TILE_K * TILE_N];
    uint64_t mbar_load;
} stages[NUM_STAGES];
```

## Key Points

1. **Producer-consumer separation**: Dedicated warp(s) for TMA, others for compute
2. **Circular buffer**: Stages wrap around with modulo indexing
3. **Barrier per stage**: Independent tracking of each buffer's state
4. **Prologue/epilogue**: Fill pipeline before mainloop, drain after

## Related Skills
- `cuda-matmul-warp-specialization`: Producer/consumer warp roles
- `cuda-matmul-barrier`: mbarrier operations
- `cuda-matmul-tma`: TMA load operations
