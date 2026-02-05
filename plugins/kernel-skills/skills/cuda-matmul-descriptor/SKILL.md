---
name: cuda-matmul-descriptor
description: Build GMMA descriptors for WGMMA instructions in CUDA matmul on Hopper+ GPUs. Use when constructing matrix descriptors for Tensor Core operations, or when encoding swizzle patterns for shared memory operands on sm90+.
---

# GMMA Descriptor Construction

WGMMA instructions read matrix operands via 64-bit descriptors that encode SMEM address, layout, and swizzle pattern.

## Descriptor Format

```
Bits [13:0]   - Base address bits [17:4] (16-byte aligned SMEM address)
Bits [15:14]  - Leading dimension mode (0=no stride, 1=16B, 2=32B, 3=64B)
Bits [29:16]  - Stride dimension byte offset
Bits [31:30]  - Reserved
Bits [45:32]  - Leading dimension byte offset
Bits [61:46]  - Reserved
Bits [63:62]  - Layout type (swizzle mode)
```

## Layout Types (Swizzle Modes)

| Value | Mode | Description |
|-------|------|-------------|
| 0 | None | No swizzle |
| 1 | 32B | 32-byte swizzle |
| 2 | 64B | 64-byte swizzle |
| 3 | 128B | **128-byte swizzle (standard)** |

## Basic Descriptor Construction

```cuda
__device__ uint64_t make_gmma_desc_b128(void* smem_ptr) {
    uint64_t desc = 0;
    uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);

    // Address field: bits [17:4] of SMEM address -> bits [13:0] of descriptor
    desc |= (addr >> 4) & 0x3FFF;

    // Layout type: B128 swizzle -> bits [63:62] = 0b11
    desc |= (uint64_t)3 << 62;

    return desc;
}
```

## Descriptor with Stride

For non-contiguous tiles or K-dimension iteration:

```cuda
// K-major layout: stride along M dimension
__device__ uint64_t make_gmma_desc_strided(
    void* smem_base,
    int leading_dim_bytes,   // Stride in bytes
    int stride_offset_bytes  // Offset for this K iteration
) {
    uint64_t desc = 0;
    uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_base);

    // Base address (must be 16B aligned)
    desc |= (addr >> 4) & 0x3FFF;

    // Leading dimension mode: encode stride as multiple of 16B
    // Mode 1=16B, 2=32B, 3=64B stride
    int ld_mode = 0;
    if (leading_dim_bytes == 16) ld_mode = 1;
    else if (leading_dim_bytes == 32) ld_mode = 2;
    else if (leading_dim_bytes == 64) ld_mode = 3;
    desc |= (uint64_t)ld_mode << 14;

    // Stride dimension offset in bytes
    desc |= ((uint64_t)stride_offset_bytes & 0x3FFF) << 16;

    // Leading dimension offset (for tiled layouts)
    desc |= ((uint64_t)leading_dim_bytes & 0x3FFF) << 32;

    // B128 swizzle
    desc |= (uint64_t)3 << 62;

    return desc;
}
```

## Pre-computed Base Descriptor

Optimize by computing base descriptor once and adding offsets:

```cuda
struct GmmaDescriptor {
    uint64_t base_desc;
    int k_stride_bytes;

    __device__ void init(void* smem_base, int tile_k, int elem_size) {
        base_desc = make_gmma_desc_b128(smem_base);
        k_stride_bytes = tile_k * elem_size;  // e.g., 64 * 2 = 128 bytes
    }

    // Get descriptor for K iteration
    __device__ uint64_t get(int k_iter) {
        // Add K offset to address field
        int k_offset_bytes = k_iter * 16 * sizeof(__nv_bfloat16);  // 16 = MMA_K
        uint64_t desc = base_desc;
        desc += (k_offset_bytes >> 4);  // Add to address bits [13:0]
        return desc;
    }
};
```

## Descriptor for A Matrix (M×K, K-major)

```cuda
// A tile: 64×64 BF16, K-major layout
// Each WGMMA iteration consumes 64×16 sub-tile

__device__ uint64_t make_desc_A(
    void* smem_a,      // Base of A tile in SMEM
    int k_iter         // Which K=16 slice (0 to TILE_K/16-1)
) {
    uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_a);

    // Offset for this K iteration: k_iter * 16 elements * 2 bytes
    int k_offset = k_iter * 16 * sizeof(__nv_bfloat16);
    addr += k_offset;

    uint64_t desc = 0;
    desc |= (addr >> 4) & 0x3FFF;
    desc |= (uint64_t)3 << 62;  // B128 swizzle

    return desc;
}
```

## Descriptor for B Matrix (K×N, K-major)

```cuda
// B tile: 64×256 BF16, K-major layout

__device__ uint64_t make_desc_B(
    void* smem_b,
    int k_iter
) {
    uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_b);

    // K-major: K is contiguous, so offset = k_iter * N * elem_size
    // But with swizzle, addressing is more complex
    int k_offset = k_iter * 16 * sizeof(__nv_bfloat16);
    addr += k_offset;

    uint64_t desc = 0;
    desc |= (addr >> 4) & 0x3FFF;
    desc |= (uint64_t)3 << 62;

    return desc;
}
```

## Complete Example: Mainloop Descriptors

```cuda
__device__ void compute_wgmma(
    void* smem_a,
    void* smem_b,
    float* accum
) {
    constexpr int TILE_K = 64;
    constexpr int MMA_K = 16;
    constexpr int K_ITERS = TILE_K / MMA_K;  // 4

    // Pre-compute base descriptors
    uint64_t desc_a_base = make_gmma_desc_b128(smem_a);
    uint64_t desc_b_base = make_gmma_desc_b128(smem_b);

    #pragma unroll
    for (int k = 0; k < K_ITERS; k++) {
        // Compute descriptor offsets for this K iteration
        int k_offset_16b = k * MMA_K * sizeof(__nv_bfloat16) / 16;

        uint64_t desc_a = desc_a_base + k_offset_16b;
        uint64_t desc_b = desc_b_base + k_offset_16b;

        // Issue WGMMA
        wgmma_m64n256k16(desc_a, desc_b, accum, /*accumulate=*/k > 0);
    }

    // Commit group
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
}
```

## Debugging Descriptors

```cuda
__device__ void print_desc(uint64_t desc) {
    printf("Descriptor: 0x%016llx\n", desc);
    printf("  Address bits [13:0]: 0x%04x -> SMEM offset 0x%05x\n",
           (int)(desc & 0x3FFF), (int)((desc & 0x3FFF) << 4));
    printf("  LD mode [15:14]: %d\n", (int)((desc >> 14) & 0x3));
    printf("  Stride [29:16]: %d bytes\n", (int)((desc >> 16) & 0x3FFF));
    printf("  LD offset [45:32]: %d bytes\n", (int)((desc >> 32) & 0x3FFF));
    printf("  Layout [63:62]: %d\n", (int)((desc >> 62) & 0x3));
}
```

## Key Points

1. **16-byte alignment**: Address field assumes 16B alignment
2. **Swizzle matching**: Descriptor layout must match TMA/SMEM swizzle
3. **K-major layout**: WGMMA expects K-major tiles (K contiguous)
4. **Offset arithmetic**: K iteration offsets add to address field
5. **Warpgroup-wide**: Same descriptor used by all 128 threads

## Related Skills
- `cuda-matmul-wgmma`: Using descriptors with WGMMA
- `cuda-matmul-swizzle`: SMEM layout for descriptor
- `cuda-matmul-tma`: TMA swizzle encoding
