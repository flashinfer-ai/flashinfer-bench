import type { CurvePoint } from "@/lib/analytics"

export type CorrectnessStats = {
  total: number
  passed: number
  incorrect: number
  runtime_error: number
  other: number
}

export type CurvesPayload = {
  nWorkloads: number
  curves: Record<string, CurvePoint[]>
  correctness: Record<string, CorrectnessStats>
}

export type WorkloadFiltersState = {
  axisRanges: Record<string, { min: number; max: number }>
  devices: string[]
  onlyPassed: boolean
}

export type SolutionFiltersState = {
  languages: string[]
  authors: string[]
  targets: string[]
  search: string
}
