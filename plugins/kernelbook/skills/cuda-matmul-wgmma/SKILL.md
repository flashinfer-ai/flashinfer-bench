---
name: cuda-matmul-wgmma
description: Write CUDA matmul kernels using WGMMA (Warp-Group MMA) instructions for Hopper+ GPUs (sm90+). Use when implementing matrix multiplication with Tensor Core operations, or when optimizing GEMM performance on H100/H800.
---

# WGMMA (Warp-Group Matrix Multiply-Accumulate)

WGMMA is the primary Tensor Core instruction for SM90+ (Hopper). It operates at warpgroup level (4 warps = 128 threads) and computes `D = A * B + D` asynchronously.

## Instruction Variants

### Common Shapes (FP16/BF16 → FP32)
```
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16   // Small tile
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16  // Medium tile
wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16  // Large tile (best throughput)
```

### Source Requirements
- **A operand**: Can be from registers or shared memory (via descriptor)
- **B operand**: Must be from shared memory (via descriptor)
- **D accumulator**: Always in registers

## Basic Usage Pattern

```cuda
#include <cuda_fp16.h>

// WGMMA requires warpgroup (128 threads) coordination
__device__ void wgmma_m64n256k16(
    uint64_t desc_a,      // GMMA descriptor for A in SMEM
    uint64_t desc_b,      // GMMA descriptor for B in SMEM
    float (&d)[64],       // Accumulator in registers (64 floats per thread for m64n256)
    bool accumulate = true
) {
    int scale_d = accumulate ? 1 : 0;

    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, %66;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(desc_a), "l"(desc_b), "n"(scale_d)
    );
}
```

## Synchronization Requirements

WGMMA is asynchronous. Must use commit/wait for correctness:

```cuda
// After issuing WGMMA instructions
asm volatile("wgmma.commit_group.sync.aligned;\n" ::);

// Before reading accumulator results
asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(0));
```

## Accumulator Layout

For m64n256k16 with FP32 accumulator, each thread holds 64 floats covering a 64×256 tile:
- Thread layout: 4 warps × 32 threads = 128 threads
- Each thread owns a strided pattern of the output tile
- Register count: 64 × 4 bytes = 256 bytes per thread

## Key Constraints

1. **Warpgroup requirement**: All 128 threads must execute WGMMA together
2. **SMEM layout**: B operand requires swizzled layout (B128 swizzle recommended)
3. **Descriptor alignment**: SMEM addresses must be 16-byte aligned
4. **K dimension**: Fixed at 16 elements (32 bytes for BF16/FP16)

## Choosing Tile Size

| Shape | Registers/Thread | Throughput | Use Case |
|-------|-----------------|------------|----------|
| m64n64k16 | 16 | Lower | Register-limited kernels |
| m64n128k16 | 32 | Medium | Balanced |
| m64n256k16 | 64 | Highest | Maximum performance |

## Related Skills
- `cuda-matmul-descriptor`: Building GMMA descriptors
- `cuda-matmul-swizzle`: SMEM layout for WGMMA
- `cuda-matmul-barrier`: Synchronization with mbarrier
