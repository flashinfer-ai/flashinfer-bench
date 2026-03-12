---
name: cuda-matmul-warp-specialization
description: Implement warp specialization for CUDA matmul with dedicated producer and consumer warps on Hopper+ GPUs. Use when separating TMA loading and WGMMA computation into different warps, or when using dynamic register reallocation on sm90+.
---

# Warp Specialization for Matmul

Warp specialization assigns different roles to warps: TMA producers handle memory loads while math consumers execute WGMMA operations. This enables overlapping memory fetches and computation and optimal register allocation based on distinct producer and consumer requirements.

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
    bool is_producer_warpgroup = (warpgroup_id == 0);
    bool is_producer_loader_warp = (warp_id == 0);  // Dedicated TMA warp
    bool is_consumer_warpgroup = (warpgroup_id == 1);  // Warpgroup 1

    if (is_producer_warpgroup) {
        producer_warpgroup(is_producer_loader_warp);
    } else if (is_consumer_warpgroup) {
        consumer_warpgroup();
    }
    // Other warps can assist or remain idle
}
```

## Dynamic Register Reallocation

Hopper allows reallocating registers between warpgroups at runtime. One thread can own up to 255 registers, and setmaxnreg can be set to a value between 24 and 256 (inclusive) at multiples of 8:

```cuda
__device__ void producer_warpgroup(bool is_loader_warp) {
    // All threads in producer warpgroup execute setmaxnreg together
    // Producer needs few registers (only TMA coordination)
    // Release registers to consumers
    asm volatile("setmaxnreg.dec.sync.aligned.u32 48;\n" ::);

    if (!is_loader_warp) return;

    // TMA operations (only loader warp participates; lane 0 issues TMA)
    // ...
}

__device__ void consumer_warpgroup() {
    // All threads in consumer warpgroup execute setmaxnreg together
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

Use this register-budget rule per SM:

`128 * (P * R_p + C * R_c) <= 65536`

Equivalent per-thread form:

`P * R_p + C * R_c <= 512`

Where `P` is producer warpgroups, `C` is consumer warpgroups, `R_p` is registers/thread for producers, and `R_c` is registers/thread for consumers.

Decision rule:
- Choose the smallest feasible producer value `R_p` first.
- Compute `R_c_max = floor((512 - P * R_p) / C)`.
- Round `R_c_max` down to a multiple of 8 and cap at 255 (setmaxnreg range constraints).

Example (`P=1`, `C=2`, `R_p=24`):
- `R_c_max = floor((512 - 24) / 2) = 244` -> use `240` (multiple of 8), yielding split `24/240/240`.

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
            mbarrier_init(&shared->mbar[s], 128);       // TMA -> consumer warpgroup
            mbarrier_init(&shared->mbar_empty[s], 1);   // Consumer lane 0 -> producer lane 0
        }
    }
    __syncthreads();

    if (warpgroup_id == 0) {
        // === PRODUCER WARPGROUP ===
        // setmaxnreg must be executed by all threads in this warpgroup
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;\n" ::);

        if (warp_id == 0) {
            // Dedicated producer warp controls pipeline
            int m_tile = blockIdx.x;
            int n_tile = blockIdx.y;
            int num_k_tiles = K / TILE_K;
            int empty_phase[NUM_STAGES] = {0};

            for (int k = 0; k < num_k_tiles; k++) {
                int stage = k % NUM_STAGES;

                // Wait for consumer to finish with this stage before reuse
                if (lane_id == 0 && k >= NUM_STAGES) {
                    mbarrier_wait(&shared->mbar_empty[stage], empty_phase[stage]);
                    empty_phase[stage] ^= 1;  // Flip phase on each stage reuse
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
        }

    } else if (warpgroup_id == 1) {
        // === CONSUMER WARPGROUP ===
        // setmaxnreg must be executed by all threads in this warpgroup
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::);

        // Initialize accumulators
        float accum[64];
        #pragma unroll
        for (int i = 0; i < 64; i++) accum[i] = 0.0f;

        int num_k_tiles = K / TILE_K;
        int load_phase[NUM_STAGES] = {0};

        for (int k = 0; k < num_k_tiles; k++) {
            int stage = k % NUM_STAGES;

            // Wait for TMA load to complete
            mbarrier_wait(&shared->mbar[stage], load_phase[stage]);
            load_phase[stage] ^= 1;  // Flip phase on each stage reuse

            // Build GMMA descriptors
            uint64_t desc_a = make_gmma_desc(shared->smem_a[stage]);
            uint64_t desc_b = make_gmma_desc(shared->smem_b[stage]);

            // Execute WGMMA (k=0 overwrite, k>0 accumulate)
            if (k == 0) {
                wgmma_m64n256k16<0>(desc_a, desc_b, accum);
            } else {
                wgmma_m64n256k16<1>(desc_a, desc_b, accum);
            }

            // Commit and signal buffer available
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
            if (lane_id == 0) {
                mbarrier_arrive(&shared->mbar_empty[stage]);
            }
        }

        // Wait for all WGMMA
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::);

        // Store results via TMA or direct store
        // ...
    }
}
```

## Barrier Phase Handling

- Track one phase bit per pipeline stage for both `mbar` and `mbar_empty`.
- Pass the current phase to `mbarrier_wait`.
- Flip the phase bit after each successful wait before reusing the same stage.

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

## Key Constraints

1. **setmaxnreg sync**: All threads in warpgroup must execute together
2. **Barrier sizing**: Match arrival count to participating threads
3. **Phase alternation**: Flip per-stage phase between consecutive stage reuses
4. **Memory ordering**: Use `fence.proxy.async` when needed between roles
5. **Single TMA thread**: Only lane 0 of producer warp issues TMA

## Related Skills
- `cuda-matmul-pipelining`: Multi-stage pipeline with producer/consumer
- `cuda-matmul-barrier`: Barrier operations for synchronization
- `cuda-matmul-wgmma`: Consumer warpgroup computation
- `cuda-matmul-tma`: Basic TMA operations
