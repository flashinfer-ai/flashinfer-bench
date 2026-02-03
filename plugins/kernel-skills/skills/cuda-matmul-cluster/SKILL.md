---
name: cuda-matmul-cluster
description: Configure and use SM clusters for CUDA matmul on Hopper+ GPUs. Use when implementing multi-CTA cooperation, or when optimizing tile scheduling and L2 locality for GEMM on sm90+.
---

# Cluster-Level Parallelism for Matmul

SM90+ introduces clusters: groups of CTAs that can share data via distributed shared memory and coordinate efficiently.

## Cluster Configuration

### Launch Configuration

```cuda
// Cluster dimensions (e.g., 2x1x1 = 2 CTAs per cluster)
constexpr int CLUSTER_M = 2;
constexpr int CLUSTER_N = 1;

cudaLaunchConfig_t config = {0};
config.blockDim = dim3(BLOCK_SIZE);
config.gridDim = dim3(num_blocks_m, num_blocks_n);

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {CLUSTER_M, CLUSTER_N, 1};

config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, kernel, args...);
```

### Cluster Intrinsics

```cuda
// Get cluster position
__device__ int cluster_id_in_grid() {
    uint32_t id;
    asm volatile("mov.u32 %0, %%clusterid;\n" : "=r"(id));
    return id;
}

__device__ dim3 cluster_dim() {
    uint32_t x, y, z;
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(x));
    asm volatile("mov.u32 %0, %%cluster_nctaid.y;\n" : "=r"(y));
    asm volatile("mov.u32 %0, %%cluster_nctaid.z;\n" : "=r"(z));
    return {x, y, z};
}

// CTA position within cluster
__device__ int cta_rank_in_cluster() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    return rank;
}
```

## Distributed Shared Memory

CTAs in a cluster can access each other's shared memory:

```cuda
// Map local SMEM address to remote CTA
__device__ void* mapa_shared_cluster(void* local_smem, int dst_cta_rank) {
    uint64_t local_addr = (uint64_t)__cvta_generic_to_shared(local_smem);
    uint64_t remote_addr;

    asm volatile(
        "mapa.shared::cluster.u64 %0, %1, %2;\n"
        : "=l"(remote_addr)
        : "l"(local_addr), "r"(dst_cta_rank)
    );

    return (void*)remote_addr;
}

// Read from remote CTA's SMEM
__device__ float read_remote(float* local_ptr, int dst_cta) {
    float* remote_ptr = (float*)mapa_shared_cluster(local_ptr, dst_cta);
    return *remote_ptr;
}
```

## Cluster Synchronization

```cuda
// Sync all CTAs in cluster
__device__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive;\n" ::);
    asm volatile("barrier.cluster.wait;\n" ::);
}

// Arrive without waiting
__device__ void cluster_arrive() {
    asm volatile("barrier.cluster.arrive;\n" ::);
}

// Wait for all arrivals
__device__ void cluster_wait() {
    asm volatile("barrier.cluster.wait;\n" ::);
}
```

## Tile Scheduling with Clusters

Clusters work on adjacent tiles for L2 locality:

```cuda
// GROUP_M scheduling: Process tiles in groups along M
constexpr int GROUP_M = 8;  // Tiles per group

__device__ void get_tile_coords(int& m_tile, int& n_tile, int M_tiles, int N_tiles) {
    int cluster_id = cluster_id_in_grid();
    int cta_rank = cta_rank_in_cluster();

    // Linear cluster index
    int linear_id = cluster_id * (CLUSTER_M * CLUSTER_N) + cta_rank;

    // GROUP_M scheduling
    int num_groups = (M_tiles + GROUP_M - 1) / GROUP_M;
    int group_id = linear_id / (GROUP_M * N_tiles);
    int in_group = linear_id % (GROUP_M * N_tiles);

    int m_in_group = in_group / N_tiles;
    n_tile = in_group % N_tiles;
    m_tile = group_id * GROUP_M + m_in_group;

    // Serpentine pattern for even better locality
    if (group_id % 2 == 1) {
        n_tile = N_tiles - 1 - n_tile;
    }
}
```

## Cluster Matmul Pattern

```cuda
// 2x1 cluster: 2 CTAs share B tile
constexpr int CLUSTER_M = 2;
constexpr int CLUSTER_N = 1;

__global__ void cluster_matmul(
    const CUtensorMap* tma_a,
    const CUtensorMap* tma_b,
    float* C,
    int M, int N, int K
) {
    extern __shared__ char smem[];

    int cta_rank = cta_rank_in_cluster();
    int m_tile, n_tile;
    get_tile_coords(m_tile, n_tile, M / TILE_M, N / TILE_N);

    // Each CTA in cluster handles different M tile, same N tile
    m_tile = m_tile * CLUSTER_M + cta_rank;

    // CTA 0 loads B tile (shared by cluster)
    // Both CTAs load their own A tile

    for (int k = 0; k < K / TILE_K; k++) {
        if (cta_rank == 0) {
            // Load B tile (will be multicast to cluster)
            tma_load_multicast(smem_b, tma_b, ...);
        }

        // Each CTA loads its own A tile
        tma_load(smem_a, tma_a, k * TILE_K, m_tile * TILE_M, ...);

        cluster_sync();  // Ensure all loads complete

        // Compute WGMMA
        // ...
    }
}
```

## Cluster Benefits for Matmul

| Benefit | Description |
|---------|-------------|
| **B tile sharing** | Load B once, use in multiple CTAs via multicast |
| **L2 locality** | Adjacent tiles in cluster hit same cache lines |
| **Reduced bandwidth** | 2x cluster = 2x less B tile bandwidth |
| **Barrier efficiency** | Hardware cluster barriers faster than global atomics |

## Cluster Size Selection

| Cluster | CTAs | Best For |
|---------|------|----------|
| 1×1 | 1 | Baseline, small matrices |
| 2×1 | 2 | **Common choice**, share B along M |
| 1×2 | 2 | Share A along N |
| 2×2 | 4 | Large tiles, share both |

## Key Points

1. **Static cluster dim**: Set at launch, cannot change dynamically
2. **SMEM alignment**: Remote SMEM access requires proper alignment
3. **Barrier scope**: Use cluster barriers for cross-CTA sync
4. **Tile assignment**: Account for cluster size in tile scheduling
5. **Max cluster size**: Hardware limit (typically 8-16 CTAs)

## Related Skills
- `cuda-matmul-tma-multicast`: TMA multicast across cluster
- `cuda-matmul-pipelining`: Pipeline with cluster coordination
- `cuda-matmul-barrier`: Cluster barrier operations
