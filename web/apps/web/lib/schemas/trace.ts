import {z} from 'zod'

// Axis schema - matching flashinfer-bench
export const ConstAxisSchema = z.object({
  type: z.literal('const'),
  value: z.number(),
  description: z.string().optional(),
})

export const VarAxisSchema = z.object({
  type: z.literal('var'),
  parent: z.string().optional(),
  description: z.string().optional(),
})

export const AxisSchema = z.union([ConstAxisSchema, VarAxisSchema])

// Tensor schema
export const TensorSchema = z.object({
  shape: z.array(z.string()).nullable(),
  dtype: z.enum([
    "float32",
    "float16",
    "bfloat16",
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1",
    "int64",
    "int32",
    "int16",
    "int8",
    "bool",
  ]),
  description: z.string().optional(),
})

// Definition schema - matching flashinfer-bench
export const DefinitionSchema = z.object({
  name: z.string(),
  op_type: z.string(), // General op type like "gemm", "attention", "rmsnorm"
  description: z.string().optional(),
  tags: z.array(z.string()).optional(),
  axes: z.record(z.string(), AxisSchema),
  inputs: z.record(z.string(), TensorSchema),
  outputs: z.record(z.string(), TensorSchema),
  reference: z.string(),  // PyTorch reference implementation
  constraints: z.array(z.string()).optional(),
})

// Solution schema - matching flashinfer-bench
export const SolutionSchema = z.object({
  name: z.string(),
  definition: z.string(), // Name of the Definition it solves
  description: z.string().optional(),
  author: z.string(),
  spec: z.object({
    language: z.enum(["python", "triton", "cuda"]),
    target_hardware: z.array(z.string()),
    entry_point: z.string(),
    dependencies: z.array(z.string()).optional(),
    build_commands: z.array(z.string()).optional(),
  }),
  sources: z.array(z.object({
    path: z.string(),
    content: z.string(),
  })),
})

// Workload input schema
export const WorkloadInputSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('random'),
    seed: z.number().optional(),
  }),
  z.object({
    type: z.literal('safetensors'),
    path: z.string(),
    tensor_key: z.string(),
  }),
  z.object({
    type: z.literal("scalar"),
    value: z.union([z.number(), z.boolean()]),
  }),
])

// Workload schema
export const WorkloadSchema = z.object({
  uuid: z.string(),
  axes: z.record(z.string(), z.number()), // Concrete values for variable axes
  inputs: z.record(z.string(), WorkloadInputSchema),
})

// Evaluation schema - matching flashinfer-bench
export const EvaluationSchema = z.object({
  status: z.enum([
    'PASSED', 'INCORRECT_SHAPE', 'INCORRECT_NUMERICAL', 'INCORRECT_DTYPE',
    'RUNTIME_ERROR', 'COMPILE_ERROR'
  ]),
  log_file: z.string(),
  correctness: z.object({
                  max_relative_error: z.number(),
                  max_absolute_error: z.number(),
                }).nullable(),
  performance: z.object({
                  latency_ms: z.number(),
                  reference_latency_ms: z.number(),
                  speedup_factor: z.number(),
                }).nullable(),
  environment: z.object({
    hardware: z.string(),
    device: z.string().optional(),
    libs: z.record(z.string(), z.string()).optional(),
  }),
  timestamp: z.string(),  // ISO 8601 timestamp
})

// Trace schema - matching flashinfer-bench
export const TraceSchema = z.object({
  definition: z.string(),           // Name of the Definition
  solution: z.string().nullable(),  // Name of the Solution (null for
                                    // workload-only traces)
  workload: WorkloadSchema,
  evaluation: EvaluationSchema.nullable(),  // Null for workload-only traces
})

// Canonical workload schema for UI
export const CanonicalWorkloadSchema = z.object({
  name: z.string(),
  description: z.string(),
  axes: z.record(z.string(), z.number()),
})

// Type exports
export type ConstAxis = z.infer<typeof ConstAxisSchema>
export type VarAxis = z.infer<typeof VarAxisSchema>
export type Axis = z.infer<typeof AxisSchema>
export type Tensor = z.infer<typeof TensorSchema>
export type Definition = z.infer<typeof DefinitionSchema>
export type Solution = z.infer<typeof SolutionSchema>
export type WorkloadInput = z.infer<typeof WorkloadInputSchema>
export type Workload = z.infer<typeof WorkloadSchema>
export type Evaluation = z.infer<typeof EvaluationSchema>
export type Trace = z.infer<typeof TraceSchema>
export type CanonicalWorkload = z.infer<typeof CanonicalWorkloadSchema>
