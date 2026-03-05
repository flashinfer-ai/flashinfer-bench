---
name: cuda-matmul-block-scaling
description: Implement block-scaled matmul with FP4/FP6/FP8 on Blackwell GPUs (sm100). Use when working with low-precision quantized models, or when leveraging native block scaling support in UMMA for inference optimization.
---

# Block Scaling for Low-Precision Matmul on Blackwell

Blackwell UMMA natively supports block scaling for FP4, FP6, and FP8 data types, enabling efficient quantized inference without separate dequantization passes.

## Supported Precisions

| Type | Bits | Range | Use Case |
|------|------|-------|----------|
| FP8 (E4M3) | 8 | ±240 | Training/inference |
| FP8 (E5M2) | 8 | ±57344 | Training gradients |
| FP6 | 6 | Limited | Inference |
| **FP4** | 4 | Limited | **Inference (4-bit quantization)** |

## Block Scaling Concept

```
Matrix A (quantized):     Scale factors:
┌─────────────────┐      ┌─────┐
│ block 0 (FP4)   │  ×   │ s0  │ (FP32/FP8)
│ block 1 (FP4)   │      │ s1  │
│ block 2 (FP4)   │      │ s2  │
│ ...             │      │ ... │
└─────────────────┘      └─────┘

Effective value = quantized_value × block_scale
```

## UMMA with Block Scaling

```cuda
// Instruction for FP8/FP6/FP4 with block scaling
// tcgen05.mma.cta_group.kind::f8f6f4 [d], a-desc, b-desc, idesc, scale-d;

__device__ void umma_fp4_scaled(
    uint32_t tmem_d,
    uint64_t desc_a,        // FP4 data descriptor
    uint64_t desc_b,        // FP4 data descriptor
    uint64_t desc_a_scale,  // A scale factors descriptor
    uint64_t desc_b_scale,  // B scale factors descriptor
    uint32_t idesc,
    bool accumulate
) {
    int scale_d = accumulate ? 1 : 0;

    // Instruction descriptor includes scale factor info
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f8f6f4 "
        "[%0], %1, %2, %3, %4;\n"
        :
        : "r"(tmem_d), "l"(desc_a), "l"(desc_b),
          "r"(idesc), "r"(scale_d)
        : "memory"
    );
}
```

## Instruction Descriptor for Scaled MMA

```cuda
// Instruction descriptor bits for block scaling
struct ScaledIdesc {
    uint32_t a_type : 4;      // A data type (FP4/FP6/FP8)
    uint32_t b_type : 4;      // B data type
    uint32_t a_scale_type : 2; // A scale factor type
    uint32_t b_scale_type : 2; // B scale factor type
    uint32_t transpose_a : 1;
    uint32_t transpose_b : 1;
    uint32_t negate_a : 1;
    uint32_t negate_b : 1;
    // ... other fields
};

__device__ uint32_t make_scaled_idesc(
    int a_dtype,  // 0=FP8_E4M3, 1=FP8_E5M2, 2=FP6, 3=FP4
    int b_dtype,
    int scale_dtype  // 0=FP32, 1=FP8
) {
    uint32_t idesc = 0;
    idesc |= (a_dtype & 0xF);
    idesc |= (b_dtype & 0xF) << 4;
    idesc |= (scale_dtype & 0x3) << 8;
    idesc |= (scale_dtype & 0x3) << 10;
    return idesc;
}
```

## Memory Layout for Block-Scaled Data

### Data Layout (FP4)

```
Tile: 128×64 elements in FP4
- Packed: 2 elements per byte
- Size: 128×64/2 = 4KB
- Layout: K-major with swizzle
```

### Scale Factor Layout

```
Block size: 32 elements along K (typical)
For 128×64 tile with K=64:
- Blocks per row: 64/32 = 2
- Total blocks: 128×2 = 256 scale factors
- Scale format: FP32 (1KB) or FP8 (256B)
```

## TMA Descriptor for Scaled Data

```cuda
// Separate TMA descriptors for data and scales
CUtensorMap create_fp4_data_desc(
    void* data_ptr,
    int M, int K_packed  // K_packed = K/2 for FP4
) {
    CUtensorMap desc{};

    uint64_t size[2] = {(uint64_t)K_packed, (uint64_t)M};
    uint64_t stride[1] = {(uint64_t)K_packed};
    uint32_t box[2] = {32, 128};  // Tile size in packed elements

    cuTensorMapEncodeTiled(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,  // FP4 packed as uint8
        2, data_ptr, size, stride, box, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_OOBFILL_ZERO
    );

    return desc;
}

CUtensorMap create_scale_desc(
    void* scale_ptr,
    int M, int num_k_blocks,
    bool is_fp8_scale
) {
    CUtensorMap desc{};

    auto dtype = is_fp8_scale ?
        CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32;

    uint64_t size[2] = {(uint64_t)num_k_blocks, (uint64_t)M};
    // ... configure for scale factor tile

    return desc;
}
```

## Complete Block-Scaled Kernel

```cuda
constexpr int BLOCK_K = 32;  // Elements per scale factor

__global__ void matmul_fp4_scaled(
    const CUtensorMap* tma_a,        // FP4 data
    const CUtensorMap* tma_a_scale,  // Scale factors
    const CUtensorMap* tma_b,
    const CUtensorMap* tma_b_scale,
    float* C,
    int M, int N, int K
) {
    extern __shared__ char smem[];

    // SMEM layout
    auto* smem_a = reinterpret_cast<uint8_t*>(smem);           // FP4 packed
    auto* smem_a_scale = reinterpret_cast<float*>(smem + ...); // Scales
    auto* smem_b = reinterpret_cast<uint8_t*>(smem + ...);
    auto* smem_b_scale = reinterpret_cast<float*>(smem + ...);

    // TMEM allocation
    __shared__ uint32_t tmem_base;
    // ... allocate

    for (int k_tile = 0; k_tile < K / TILE_K; k_tile++) {
        // Load data tiles (FP4 packed)
        tma_load_2d(smem_a, tma_a, ...);
        tma_load_2d(smem_b, tma_b, ...);

        // Load scale factor tiles
        tma_load_2d(smem_a_scale, tma_a_scale, ...);
        tma_load_2d(smem_b_scale, tma_b_scale, ...);

        // Wait for loads
        // ...

        // UMMA with block scaling
        uint32_t idesc = make_scaled_idesc(3, 3, 0);  // FP4, FP4, FP32 scales

        for (int k = 0; k < TILE_K / 16; k++) {
            uint64_t desc_a = make_smem_desc(smem_a, k);
            uint64_t desc_b = make_smem_desc(smem_b, k);

            // Scales are applied automatically by hardware
            umma_fp4_scaled(tmem_base, desc_a, desc_b,
                           desc_a_scale, desc_b_scale,
                           idesc, k > 0);
        }
    }

    // Epilogue: results are already in FP32
    // ...
}
```

## Scale Factor Configurations

| Config | A Scale | B Scale | Notes |
|--------|---------|---------|-------|
| Per-tensor | 1 | 1 | Simplest, lowest accuracy |
| Per-row/col | M | N | Good balance |
| **Per-block** | M×K/32 | K/32×N | **Best accuracy** |

## Quantization Helper

```cuda
// Host-side quantization to FP4 with block scales
void quantize_to_fp4(
    const float* input,
    uint8_t* output_packed,
    float* scales,
    int M, int K,
    int block_size = 32
) {
    int num_blocks_k = K / block_size;

    for (int m = 0; m < M; m++) {
        for (int kb = 0; kb < num_blocks_k; kb++) {
            // Find max in block for scale
            float max_val = 0;
            for (int k = 0; k < block_size; k++) {
                max_val = fmaxf(max_val, fabsf(input[m * K + kb * block_size + k]));
            }

            // Scale to FP4 range (±6 for E2M1)
            float scale = max_val / 6.0f;
            scales[m * num_blocks_k + kb] = scale;

            // Quantize block
            for (int k = 0; k < block_size; k += 2) {
                float v0 = input[m * K + kb * block_size + k] / scale;
                float v1 = input[m * K + kb * block_size + k + 1] / scale;

                // Clamp and convert to FP4
                uint8_t q0 = float_to_fp4(v0);
                uint8_t q1 = float_to_fp4(v1);

                // Pack two FP4 values into one byte
                output_packed[(m * K + kb * block_size + k) / 2] = (q1 << 4) | q0;
            }
        }
    }
}
```

## Performance Considerations

| Precision | Memory BW | Compute | Accuracy |
|-----------|-----------|---------|----------|
| FP16 | 1× | 1× | Baseline |
| FP8 | **2×** | **2×** | Good |
| FP4 | **4×** | **4×** | Moderate |

FP4 with block scaling achieves ~90% of FP16 accuracy on many LLM tasks.

## Key Points

1. **Native support**: No separate dequant kernel needed
2. **Scale descriptor**: Scales loaded via separate TMA
3. **Block alignment**: Block size typically 32 elements
4. **FP32 accumulator**: Results accumulated in FP32 despite low-precision inputs
5. **Inference focus**: Primary use case is quantized model inference

## Related Skills
- `cuda-matmul-umma`: Base UMMA operations
- `cuda-matmul-tma`: Loading packed FP4 data
- `cuda-matmul-tmem`: FP32 accumulator storage
