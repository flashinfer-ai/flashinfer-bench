---
name: cuda-matmul-warp-specialization
description: Implement warp specialization for CUDA matmul with dedicated producer and consumer warps on Hopper+ GPUs. Use when separating TMA loading from WGMMA computation, or when using dynamic register reallocation on sm90+.
---

# Warp Specialization for Matmul

Warp specialization assigns different roles to warps: TMA producers handle memory loads while math consumers execute WGMMA. This enables true overlap and optimal register allocation.

## Role Assignment

```cuda
// Block configuration: 256 threads = 8 warps
// Warpgroup = 4 warps (128 threads)
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;  // 8 warps

__device__ void kernel_entry() {
    int warp_id = threadIdx.x / 32;
    int warpgroup_id = warp_id / 4;  // 0 or 1

    // Role assignment
    bool is_producer = (warpgroup_id == 0 && warp_id == 0);  // Warp 0 only
    bool is_consumer = (warpgroup_id == 1);  // Warps 4-7 (warpgroup 1)

    if (is_producer) {
        producer_warp();
    } else if (is_consumer) {
        consumer_warpgroup();
    }
    // Other warps can assist or remain idle
}
```

## Dynamic Register Reallocation

Hopper allows reallocating registers between warpgroups at runtime:

```cuda
__device__ void producer_warp() {
    // Producer needs few registers (only TMA coordination)
    // Release registers to consumers
    asm volatile("setmaxnreg.dec.sync.aligned.u32 48;\n" ::);

    // TMA operations (minimal register usage)
    // ...
}

__device__ void consumer_warpgroup() {
    // Consumer needs many registers for accumulators
    // Acquire extra registers from producers
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::);

    // WGMMA operations (high register usage for 64 FP32 accumulators)
    // ...
}
```

## Register Budget

| Role | Typical Registers | Purpose |
|------|------------------|---------|
| Producer | 40-48 | TMA descriptors, barriers, loop vars |
| Consumer | 200-232 | Accumulators (64×4B), descriptors |

For m64n256k16 WGMMA: 64 FP32 accumulators = 256 bytes = 64 registers per thread

## Complete Warp-Specialized Kernel

```cuda
__global__ void matmul_warp_specialized(
    const CUtensorMap* __restrict__ tma_a,
    const CUtensorMap* __restrict__ tma_b,
    CUtensorMap* __restrict__ tma_c,
    int M, int N, int K
) {
    extern __shared__ char smem[];
    auto* shared = reinterpret_cast<SharedStorage*>(smem);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warpgroup_id = warp_id / 4;

    // Initialize barriers (single thread)
    if (threadIdx.x == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&shared->mbar[s], 128);  // Consumer warpgroup size
        }
    }
    __syncthreads();

    if (warpgroup_id == 0 && warp_id == 0) {
        // === PRODUCER WARP ===
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;\n" ::);

        int m_tile = blockIdx.x;
        int n_tile = blockIdx.y;
        int num_k_tiles = K / TILE_K;

        for (int k = 0; k < num_k_tiles; k++) {
            int stage = k % NUM_STAGES;

            // Wait for consumer to finish with this buffer
            if (k >= NUM_STAGES) {
                mbarrier_wait(&shared->mbar_empty[stage]);
            }

            if (lane_id == 0) {
                // Declare expected transaction bytes
                mbarrier_arrive_expect_tx(&shared->mbar[stage],
                    TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2);

                // Issue TMA loads
                tma_load_2d(shared->smem_a[stage], tma_a,
                    k * TILE_K, m_tile * TILE_M, &shared->mbar[stage]);
                tma_load_2d(shared->smem_b[stage], tma_b,
                    n_tile * TILE_N, k * TILE_K, &shared->mbar[stage]);
            }
        }

        // Signal producer done
        if (lane_id == 0) {
            shared->producer_done = true;
        }

    } else if (warpgroup_id == 1) {
        // === CONSUMER WARPGROUP ===
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::);

        // Initialize accumulators
        float accum[64];
        #pragma unroll
        for (int i = 0; i < 64; i++) accum[i] = 0.0f;

        int num_k_tiles = K / TILE_K;

        for (int k = 0; k < num_k_tiles; k++) {
            int stage = k % NUM_STAGES;

            // Wait for TMA load to complete
            mbarrier_wait(&shared->mbar[stage]);

            // Build GMMA descriptors
            uint64_t desc_a = make_gmma_desc(shared->smem_a[stage]);
            uint64_t desc_b = make_gmma_desc(shared->smem_b[stage]);

            // Execute WGMMA
            wgmma_m64n256k16(desc_a, desc_b, accum, k > 0);

            // Commit and signal buffer available
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
            mbarrier_arrive(&shared->mbar_empty[stage]);
        }

        // Wait for all WGMMA
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::);

        // Store results via TMA or direct store
        // ...
    }
}
```

## Named Barriers for Coordination

```cuda
// Named barrier for warpgroup synchronization
__device__ void warpgroup_sync(int barrier_id) {
    asm volatile("bar.sync %0, 128;\n" :: "r"(barrier_id));
}

// Arrive without waiting
__device__ void warpgroup_arrive(int barrier_id) {
    asm volatile("bar.arrive %0, 128;\n" :: "r"(barrier_id));
}
```

## Advantages of Warp Specialization

1. **Register efficiency**: Producers release registers to consumers
2. **True overlap**: TMA and WGMMA run on different warps simultaneously
3. **Simplified control**: Each role has single-purpose code path
4. **Better occupancy**: Lower average register usage per block

## Key Points

1. **setmaxnreg sync**: All threads in warpgroup must execute together
2. **Barrier sizing**: Match arrival count to participating threads
3. **Memory ordering**: Use `fence.proxy.async` when needed between roles
4. **Single TMA thread**: Only lane 0 of producer warp issues TMA

## Related Skills
- `cuda-matmul-pipelining`: Multi-stage pipeline with producer/consumer
- `cuda-matmul-barrier`: Barrier operations for synchronization
- `cuda-matmul-wgmma`: Consumer warpgroup computation
