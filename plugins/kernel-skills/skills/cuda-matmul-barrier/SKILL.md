---
name: cuda-matmul-barrier
description: Use mbarrier and fence operations for synchronization in CUDA matmul on Hopper+ GPUs. Use when coordinating TMA loads with WGMMA computation, or when implementing producer-consumer patterns on sm90+.
---

# Barrier Synchronization for Matmul

SM90+ introduces hardware mbarrier for efficient async synchronization, essential for TMA and warp-specialized kernels.

## mbarrier Basics

```cuda
// mbarrier is a 64-bit shared memory object
__shared__ uint64_t mbar;

// Convert to shared memory address
__device__ uint64_t mbar_addr(uint64_t* mbar) {
    return (uint64_t)__cvta_generic_to_shared(mbar);
}
```

## Core Operations

### Initialize

```cuda
__device__ void mbarrier_init(uint64_t* mbar, int expected_arrivals) {
    uint64_t addr = mbar_addr(mbar);
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;\n"
        :
        : "l"(addr), "r"(expected_arrivals)
    );
}
```

### Arrive (Signal Completion)

```cuda
// Simple arrive (decrement expected count)
__device__ void mbarrier_arrive(uint64_t* mbar) {
    uint64_t addr = mbar_addr(mbar);
    asm volatile(
        "mbarrier.arrive.shared.b64 _, [%0];\n"
        :
        : "l"(addr)
        : "memory"
    );
}

// Arrive with expected transaction bytes (for TMA)
__device__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint64_t addr = mbar_addr(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        :
        : "l"(addr), "r"(tx_bytes)
        : "memory"
    );
}
```

### Wait

```cuda
// Phase-based wait (flip-flop pattern)
__device__ void mbarrier_wait(uint64_t* mbar, int phase) {
    uint64_t addr = mbar_addr(mbar);

    // Spin until phase completes
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
        "@!P1 bra WAIT_LOOP;\n"
        "}\n"
        :
        : "l"(addr), "r"(phase)
        : "memory"
    );
}

// Alternative: blocking wait
__device__ void mbarrier_wait_blocking(uint64_t* mbar, int phase) {
    uint64_t addr = mbar_addr(mbar);
    asm volatile(
        "mbarrier.wait.parity.shared.b64 [%0], %1;\n"
        :
        : "l"(addr), "r"(phase)
        : "memory"
    );
}
```

## Phase Tracking Pattern

mbarrier uses phase bit (0 or 1) to distinguish consecutive uses:

```cuda
struct BarrierPhase {
    uint64_t* mbar;
    int phase;

    __device__ void wait() {
        mbarrier_wait(mbar, phase);
    }

    __device__ void arrive() {
        mbarrier_arrive(mbar);
    }

    __device__ void flip_phase() {
        phase ^= 1;
    }
};

// Usage in mainloop
BarrierPhase barrier = {&mbar, 0};

for (int k = 0; k < num_k_tiles; k++) {
    barrier.wait();       // Wait for previous arrivals
    // ... do work ...
    barrier.arrive();     // Signal completion
    barrier.flip_phase(); // Alternate phase for next iteration
}
```

## TMA Transaction Counting

TMA uses mbarrier transaction counting for completion tracking:

```cuda
// Before TMA load
size_t expected_bytes = tile_m * tile_k * sizeof(__nv_bfloat16);
mbarrier_arrive_expect_tx(&mbar, expected_bytes);

// TMA load (hardware increments transaction count)
tma_load_2d(smem, tma_desc, coord_x, coord_y, &mbar);

// TMA hardware signals mbarrier when bytes arrive
// Consumer can wait:
mbarrier_wait(&mbar, phase);
```

## Fence Operations

Memory ordering between async operations:

```cuda
// Fence for async proxy (TMA stores)
__device__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\n" :: );
}

// Fence for shared memory visibility
__device__ void fence_view_async_shared() {
    asm volatile("fence.view_async.shared::cta;\n" :: );
}

// Full memory fence
__device__ void fence_acq_rel() {
    asm volatile("fence.acq_rel.gpu;\n" :: );
}
```

## Multi-Stage Barrier Array

```cuda
constexpr int NUM_STAGES = 4;

struct PipelineBarriers {
    uint64_t load_complete[NUM_STAGES];   // TMA -> Math
    uint64_t compute_complete[NUM_STAGES]; // Math -> TMA (buffer reuse)

    __device__ void init(int math_threads) {
        if (threadIdx.x == 0) {
            for (int s = 0; s < NUM_STAGES; s++) {
                mbarrier_init(&load_complete[s], math_threads);
                mbarrier_init(&compute_complete[s], 1);  // Single producer
            }
        }
        __syncthreads();
    }
};
```

## Cluster Barrier (Cross-CTA)

For cluster-level synchronization:

```cuda
// Arrive on remote CTA's barrier
__device__ void cluster_arrive_relaxed(uint64_t* mbar, int dst_cta) {
    uint64_t smem_addr = mbar_addr(mbar);
    uint64_t remote_addr;

    // Map to remote CTA's shared memory
    asm volatile(
        "mapa.shared::cluster.u64 %0, %1, %2;\n"
        : "=l"(remote_addr)
        : "l"(smem_addr), "r"(dst_cta)
    );

    asm volatile(
        "mbarrier.arrive.relaxed.cluster.shared.b64 _, [%0];\n"
        :
        : "l"(remote_addr)
        : "memory"
    );
}
```

## Common Patterns

### Producer-Consumer

```cuda
// Producer (TMA warp)
mbarrier_arrive_expect_tx(&mbar_load[stage], bytes);
tma_load(..., &mbar_load[stage]);

// Consumer (Math warps)
mbarrier_wait(&mbar_load[stage], phase);
// ... compute ...
mbarrier_arrive(&mbar_empty[stage]);  // Signal buffer available

// Producer waits for buffer
mbarrier_wait(&mbar_empty[stage], phase);
```

### Warpgroup Sync

```cuda
// All 128 threads in warpgroup must arrive
__device__ void warpgroup_barrier() {
    asm volatile("bar.sync 0, 128;\n" ::);
}
```

## Key Points

1. **Phase alternation**: Must flip phase bit between consecutive uses
2. **Expected arrivals**: Init count must match actual arrivals
3. **Transaction bytes**: Must exactly match TMA transfer size
4. **Single init**: Only one thread initializes each mbarrier
5. **Memory ordering**: Use fences when mixing async operations

## Related Skills
- `cuda-matmul-tma`: TMA with mbarrier completion
- `cuda-matmul-warp-specialization`: Producer-consumer roles
- `cuda-matmul-pipelining`: Multi-stage barrier management
