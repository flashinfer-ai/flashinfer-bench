---
name: cuda-matmul-swizzle
description: Implement shared memory swizzle patterns for CUDA matmul to avoid bank conflicts on Hopper+ GPUs. Use when designing SMEM layouts for TMA and WGMMA, or when optimizing shared memory access patterns for GEMM.
---

# Shared Memory Swizzle for Matmul

Swizzling XORs address bits to distribute accesses across memory banks, eliminating bank conflicts for strided access patterns common in matmul.

## B128 Swizzle (Recommended for SM90+)

B128 swizzle is the standard for WGMMA and TMA on Hopper:

```cuda
template<int BBits = 3, int MBase = 4, int SShift = 3>
struct Swizzle {
    // BBits=3: 8 rows involved in swizzle (2^3)
    // MBase=4: Start XOR from bit 4 (16-byte aligned)
    // SShift=3: XOR with bits shifted by 3

    __device__ __host__ static int apply(int offset) {
        // Extract bits [MBase, MBase+BBits) and [MBase+SShift, MBase+SShift+BBits)
        int bit_mbase = (offset >> MBase) & ((1 << BBits) - 1);
        int bit_sshift = (offset >> (MBase + SShift)) & ((1 << BBits) - 1);
        // XOR them back into position
        return offset ^ (bit_mbase ^ bit_sshift) << MBase;
    }
};

// B128 swizzle parameters
using SwizzleB128 = Swizzle<3, 4, 3>;  // 128-byte swizzle pattern
```

## How Swizzle Works

For a 128×64 BF16 tile (128 rows × 64 cols × 2 bytes = 16KB):

```
Original linear layout:
Row 0: [col 0-63]  -> bytes 0-127
Row 1: [col 0-63]  -> bytes 128-255
Row 2: [col 0-63]  -> bytes 256-383
...

After B128 swizzle:
Row 0: [col 0-63]  -> bytes 0-127      (no change, row%8=0)
Row 1: [col 0-63]  -> bytes 128-255 XOR (1<<4) = different banks
Row 2: [col 0-63]  -> bytes 256-383 XOR (2<<4) = different banks
...
```

## SMEM Allocation with Swizzle

```cuda
// Tile dimensions
constexpr int TILE_M = 128;
constexpr int TILE_K = 64;
constexpr int ELEM_SIZE = sizeof(__nv_bfloat16);  // 2 bytes

// Swizzle parameters for B128
constexpr int SWIZZLE_BITS = 3;   // 8 rows in pattern
constexpr int SWIZZLE_BASE = 4;   // 16-byte alignment
constexpr int SWIZZLE_SHIFT = 3;

// SMEM layout helper
__device__ int swizzle_offset(int row, int col) {
    int linear = row * TILE_K + col;
    int byte_offset = linear * ELEM_SIZE;

    // Apply B128 swizzle
    int m = (byte_offset >> SWIZZLE_BASE) & ((1 << SWIZZLE_BITS) - 1);
    int s = (byte_offset >> (SWIZZLE_BASE + SWIZZLE_SHIFT)) & ((1 << SWIZZLE_BITS) - 1);
    int swizzled = byte_offset ^ ((m ^ s) << SWIZZLE_BASE);

    return swizzled / ELEM_SIZE;
}

// Access pattern
__shared__ __nv_bfloat16 smem_a[TILE_M * TILE_K];

// To access element at (row, col):
__nv_bfloat16 val = smem_a[swizzle_offset(row, col)];
```

## TMA Swizzle Encoding

When creating TMA descriptors, specify swizzle mode:

```cuda
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
    CU_TENSOR_MAP_SWIZZLE_128B,    // <-- B128 swizzle
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_OOBFILL_ZERO
);
```

TMA automatically applies swizzle during load, so SMEM receives swizzled data.

## GMMA Descriptor Swizzle

GMMA descriptors encode the swizzle pattern:

```cuda
// Descriptor layout_type field for B128 swizzle
constexpr int LAYOUT_TYPE_SWIZZLE_128B = 3;  // bits [62:61] = 0b11

uint64_t make_gmma_desc(void* smem_ptr) {
    uint64_t desc = 0;
    uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);

    desc |= (addr & 0x3FFFF) >> 4;              // Address bits [17:4] -> [13:0]
    desc |= (uint64_t)LAYOUT_TYPE_SWIZZLE_128B << 62;  // Swizzle mode
    // ... other fields

    return desc;
}
```

## Swizzle Modes Comparison

| Mode | Pattern | Tile Width | Best For |
|------|---------|------------|----------|
| None | Identity | Any | Debug |
| B32 | XOR 32B | 32+ bytes | Narrow tiles |
| B64 | XOR 64B | 64+ bytes | Medium tiles |
| B128 | XOR 128B | 128+ bytes | **WGMMA (standard)** |

## Bank Conflict Analysis

Without swizzle (32 banks, 4-byte granularity):
```
Thread 0 reads row 0, col 0 -> bank 0
Thread 1 reads row 1, col 0 -> bank 0  // CONFLICT!
Thread 2 reads row 2, col 0 -> bank 0  // CONFLICT!
...
```

With B128 swizzle:
```
Thread 0 reads row 0, col 0 -> bank 0
Thread 1 reads row 1, col 0 -> bank 2  // Different bank
Thread 2 reads row 2, col 0 -> bank 4  // Different bank
...
```

## Key Points

1. **TMA + WGMMA consistency**: Both must use same swizzle mode
2. **Alignment**: SMEM allocation must be 128-byte aligned for B128
3. **Tile size constraint**: Tile width must be ≥ swizzle width (128B for B128)
4. **K-major layout**: WGMMA expects K-major tiles in SMEM

## Related Skills
- `cuda-matmul-tma`: TMA with swizzle encoding
- `cuda-matmul-descriptor`: GMMA descriptor swizzle bits
- `cuda-matmul-wgmma`: WGMMA operand requirements
