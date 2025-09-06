import { z } from "zod"

export const ModuleTypeSchema = z.enum(["block", "layer"])

export const ModuleSchema = z.object({
  count: z.number(),
  parent: z.string().optional(),
  type: ModuleTypeSchema,
  definition: z.string().optional(),
})

export const ModelSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  modules: z.record(z.string(), ModuleSchema),
})

export const ModelHierarchySchema: z.ZodType<{
  name: string
  type: z.infer<typeof ModuleTypeSchema>
  definition?: string
  children?: any[]
}> = z.object({
  name: z.string(),
  type: ModuleTypeSchema,
  definition: z.string().optional(),
  children: z.array(z.lazy(() => ModelHierarchySchema)).optional(),
})

export type Module = z.infer<typeof ModuleSchema>
export type Model = z.infer<typeof ModelSchema>
export type ModelHierarchy = z.infer<typeof ModelHierarchySchema>
export type ModuleType = z.infer<typeof ModuleTypeSchema>
