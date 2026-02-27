---
name: cuda-matmul-tmem-load
description: Move data to/from Tensor Memory (TMEM) in CUDA matmul on Blackwell GPUs (sm100). Use when copying UMMA results from TMEM to registers for epilogue, or when pre-loading operands into TMEM.
---

# TMEM Data Movement for Blackwell

After UMMA computation, results must be copied from TMEM to registers for post-processing (epilogue). TMEM also supports loading operand A for certain UMMA variants.

## Data Movement Instructions

| Instruction | Direction | Description |
|-------------|-----------|-------------|
| `tcgen05.ld` | TMEM → Registers | Load accumulator for epilogue |
| `tcgen05.st` | Registers → TMEM | Store data to TMEM |
| `tcgen05.cp` | SMEM → TMEM | Copy from shared memory |

## TMEM Load (tcgen05.ld)

### Instruction Format

```
tcgen05.ld.sync.aligned.shape.num.b32 r, [taddr];

.shape = { .16x64b, .16x128b, .16x256b, .32x32b }
.num   = { .x1, .x2, .x4, .x8, .x16, .x32, .x64, .x128 }
```

### Shape Options

| Shape | Lanes | Bits/Lane | Total per Instr |
|-------|-------|-----------|-----------------|
| 16x64b | 16 | 64 | 128 bytes |
| 16x128b | 16 | 128 | 256 bytes |
| 16x256b | 16 | 256 | 512 bytes |
| **32x32b** | 32 | 32 | **128 bytes** |

### Basic Load Operation

```cuda
// 32x32b: Each thread in warp loads one 32-bit value
__device__ void tmem_load_32x32b_x1(uint32_t tmem_addr, uint32_t& dst) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
        : "=r"(dst)
        : "r"(tmem_addr)
    );
}

// x2: Each thread loads 2 values (2 columns)
__device__ void tmem_load_32x32b_x2(uint32_t tmem_addr, uint32_t& dst0, uint32_t& dst1) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        : "r"(tmem_addr)
    );
}
```

### Lane Access Restrictions

Each warp can only access 32 consecutive lanes:
- Warp 0: lanes 0-31
- Warp 1: lanes 32-63
- Warp 2: lanes 64-95
- Warp 3: lanes 96-127

```cuda
// For 128-lane accumulator, need all 4 warps (1 warpgroup)
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;

// Base lane for this warp
int base_lane = warp_id * 32;

// TMEM address for this thread
// Address format: [31:16]=lane, [15:0]=column
uint32_t my_addr = tmem_base + ((base_lane + lane_id) << 16) + column;
```

## Loading Full Accumulator

For 128×256 FP32 accumulator after UMMA:

```cuda
__device__ void load_accumulator_128x256(
    uint32_t tmem_base,
    float* accum  // Output: registers
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp handles 32 lanes (rows of output)
    int base_lane = warp_id * 32;
    int my_lane = base_lane + lane_id;

    // Load 256 columns for this lane
    // Using .x8 to load 8 floats at a time
    #pragma unroll
    for (int col_base = 0; col_base < 256; col_base += 8) {
        uint32_t addr = tmem_base + (my_lane << 16) + col_base;

        // Load 8 consecutive columns
        uint32_t v[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
            : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]),
              "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
            : "r"(addr)
        );

        // Store to register array
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            accum[col_base + i] = __uint_as_float(v[i]);
        }
    }
}
```

## TMEM Store (tcgen05.st)

```cuda
// Store from registers to TMEM
__device__ void tmem_store_32x32b_x1(uint32_t tmem_addr, uint32_t src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
        :
        : "r"(tmem_addr), "r"(src)
        : "memory"
    );
}
```

## SMEM to TMEM Copy (tcgen05.cp)

For loading A operand into TMEM:

```cuda
// Copy from SMEM to TMEM
__device__ void smem_to_tmem_copy(
    uint32_t tmem_addr,
    void* smem_ptr,
    int num_bytes
) {
    uint64_t smem_addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);

    asm volatile(
        "tcgen05.cp.cta_group::1.sync.aligned.b128 [%0], [%1];\n"
        :
        : "r"(tmem_addr), "l"(smem_addr)
        : "memory"
    );
}
```

## Complete Epilogue Pattern

```cuda
__device__ void epilogue_store(
    uint32_t tmem_base,
    float* C_gmem,
    int M, int N,
    int tile_m, int tile_n,
    float alpha, float beta
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp handles 32 rows
    int row_base = tile_m * 128 + warp_id * 32;
    int my_row = row_base + lane_id;

    if (my_row < M) {
        // Load accumulator for this row
        float accum[256];

        int my_lane = warp_id * 32 + lane_id;
        uint32_t base_addr = tmem_base + (my_lane << 16);

        // Load all columns
        #pragma unroll
        for (int c = 0; c < 256; c += 8) {
            uint32_t addr = base_addr + c;
            uint32_t v[8];

            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]),
                  "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                : "r"(addr)
            );

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                accum[c + i] = __uint_as_float(v[i]);
            }
        }

        // Apply alpha scaling and store to global
        int col_base = tile_n * 256;
        #pragma unroll
        for (int c = 0; c < 256; c++) {
            int col = col_base + c;
            if (col < N) {
                float result = alpha * accum[c];
                if (beta != 0.0f) {
                    result += beta * C_gmem[my_row * N + col];
                }
                C_gmem[my_row * N + col] = result;
            }
        }
    }
}
```

## Vectorized Global Store

```cuda
// Use float4 for coalesced stores
__device__ void store_row_vectorized(
    float* accum,    // 256 floats
    float* C_row,    // Global memory row pointer
    int N_valid      // Valid columns
) {
    #pragma unroll
    for (int c = 0; c < 256; c += 4) {
        if (c + 4 <= N_valid) {
            float4 val = make_float4(accum[c], accum[c+1], accum[c+2], accum[c+3]);
            reinterpret_cast<float4*>(C_row)[c / 4] = val;
        }
    }
}
```

## Key Points

1. **Warp-wide operation**: `tcgen05.ld` is warp-synchronized
2. **Lane restrictions**: Each warp accesses only 32 lanes (use warpgroup for full tile)
3. **Same base address**: All threads in warp use same TMEM base address
4. **Must copy out**: No compute on TMEM - copy to registers first
5. **Vectorization**: Use larger `.num` values for efficiency

## Related Skills
- `cuda-matmul-tmem`: TMEM allocation and addressing
- `cuda-matmul-umma`: UMMA producing TMEM results
- `cuda-matmul-warp-specialization`: Warpgroup coordination for epilogue
