import { Trace, Solution } from "./schemas"

export type WorkloadGroupId = string

export type CurvePoint = { p: number; percent: number }

export type CorrectnessSummary = {
  total: number
  passed: number
  incorrect: number
  runtime_error: number
  other: number
}

export type CurvesResponse = {
  nWorkloads: number
  curves: Record<string, CurvePoint[]>
  correctness: Record<string, CorrectnessSummary>
}

export type AuthorCurvesResponse = {
  curves: Record<string, CurvePoint[]>
  comparisonCounts: Record<string, number>
  totalComparisons: number
}

export type AuthorCorrectnessResponse = {
  stats: Array<CorrectnessSummary & { author: string }>
  totals: CorrectnessSummary
}

type Grouped = Map<WorkloadGroupId, Trace[]>

export type BaselineConfig = {
  default?: string
  devices?: Record<string, string>
}

export function groupIdForTrace(t: Trace): WorkloadGroupId {
  return t.workload?.uuid || ""
}

function deviceForTrace(t: Trace): string | null {
  return (
    t.evaluation?.environment?.device ||
    t.evaluation?.environment?.hardware ||
    null
  )
}

export function buildWorkloadGroups(traces: Trace[]): Grouped {
  const groups: Grouped = new Map()
  for (const trace of traces) {
    if (!trace || !trace.evaluation) continue
    const id = groupIdForTrace(trace)
    if (!groups.has(id)) groups.set(id, [])
    groups.get(id)!.push(trace)
  }
  return groups
}

export function solutionMap(solutions: Solution[]): Map<string, Solution> {
  const map = new Map<string, Solution>()
  for (const solution of solutions) {
    map.set(solution.name, solution)
  }
  return map
}

export function computeCorrectnessSummaryForSolutions(traces: Trace[], solutions: Solution[]): Record<string, CorrectnessSummary> {
  const solMap = solutionMap(solutions)
  const summary: Record<string, CorrectnessSummary> = {}
  for (const solution of solMap.values()) {
    summary[solution.name] = { total: 0, passed: 0, incorrect: 0, runtime_error: 0, other: 0 }
  }

  for (const trace of traces) {
    if (!trace.solution) continue
    const record = summary[trace.solution]
    if (!record) continue

    record.total += 1
    const status = trace.evaluation?.status
    if (status === "PASSED") record.passed += 1
    else if (status === "INCORRECT_SHAPE" || status === "INCORRECT_NUMERICAL" || status === "INCORRECT_DTYPE") record.incorrect += 1
    else if (status === "RUNTIME_ERROR" || status === "COMPILE_ERROR" || status === "TIMEOUT") record.runtime_error += 1
    else record.other += 1
  }

  return summary
}

function baselineForDevice(device: string, baseline?: BaselineConfig): string | null {
  if (!baseline) return null
  if (baseline.devices && baseline.devices[device]) return baseline.devices[device]
  return baseline.default ?? null
}

type SolutionGroupRatios = {
  groupRatios: Map<WorkloadGroupId, Map<string, number>>
  nWorkloads: number
  baselineNames: Set<string>
}

function computeSolutionGroupRatios(params: {
  traces: Trace[]
  solutions: Solution[]
  baseline?: BaselineConfig
}): SolutionGroupRatios {
  const { traces, solutions, baseline } = params
  const solMap = solutionMap(solutions)
  const groups = buildWorkloadGroups(traces)
  const baselineNames = new Set<string>()
  const groupRatios: Map<WorkloadGroupId, Map<string, number>> = new Map()

  if (baseline?.default) baselineNames.add(baseline.default)
  if (baseline?.devices) {
    for (const value of Object.values(baseline.devices)) baselineNames.add(value)
  }

  for (const [workloadId, groupTraces] of groups) {
    if (groupTraces.length === 0) continue

    const example = groupTraces[0]
    const device = deviceForTrace(example) || "unknown"
    const baselineName = baselineForDevice(device, baseline)
    if (!baselineName) continue

    const baselineTrace = groupTraces.find((trace) => trace.solution === baselineName)
    const baselineLatency = baselineTrace?.evaluation?.performance?.latency_ms
    if (typeof baselineLatency !== "number" || baselineLatency <= 0) continue

    for (const trace of groupTraces) {
      const solutionName = trace.solution
      if (!solutionName) continue
      if (!solMap.has(solutionName)) continue
      if (solutionName === baselineName) continue
      const candidateLatency = trace.evaluation?.performance?.latency_ms
      if (typeof candidateLatency !== "number" || candidateLatency <= 0) continue
      if (!groupRatios.has(workloadId)) groupRatios.set(workloadId, new Map())
      groupRatios.get(workloadId)!.set(solutionName, baselineLatency / candidateLatency)
    }
  }

  return {
    groupRatios,
    nWorkloads: groupRatios.size,
    baselineNames,
  }
}

export function computeFastPCurvesForSolutions(params: {
  traces: Trace[]
  solutions: Solution[]
  baseline?: BaselineConfig
  sampleCount?: number
}): { curves: Record<string, CurvePoint[]>; nWorkloads: number } {
  const { traces, solutions, baseline, sampleCount = 300 } = params
  const solMap = solutionMap(solutions)
  const { groupRatios, nWorkloads, baselineNames } = computeSolutionGroupRatios({ traces, solutions, baseline })
  const samplePoints: number[] = []
  for (let i = 0; i < sampleCount; i++) samplePoints.push(i / (sampleCount - 1))

  const curves: Record<string, CurvePoint[]> = {}
  for (const [solutionName] of solMap) {
    if (baselineNames.has(solutionName)) continue
    const ratios: number[] = []
    for (const ratiosForGroup of groupRatios.values()) {
      ratios.push(ratiosForGroup.get(solutionName) ?? 0)
    }
    ratios.sort((a, b) => a - b)
    const points: CurvePoint[] = []
    for (const p of samplePoints) {
      let lo = 0
      let hi = ratios.length
      while (lo < hi) {
        const mid = (lo + hi) >>> 1
        if (ratios[mid] < p) lo = mid + 1
        else hi = mid
      }
      const count = ratios.length - lo
      const percent = nWorkloads > 0 ? (count / nWorkloads) * 100 : 0
      points.push({ p, percent })
    }
    curves[solutionName] = points
  }

  return { curves, nWorkloads }
}

export function computeFastPCurves(params: {
  traces: Trace[]
  solutions: Solution[]
  baseline?: BaselineConfig
  sampleCount?: number
}): CurvesResponse {
  const { traces, solutions, baseline, sampleCount } = params
  const correctness = computeCorrectnessSummaryForSolutions(traces, solutions)
  const { curves, nWorkloads } = computeFastPCurvesForSolutions({ traces, solutions, baseline, sampleCount })
  return { curves, nWorkloads, correctness }
}

export function computeFastPCurvesForAuthors(params: {
  datasets: Array<{
    traces: Trace[]
    solutions: Solution[]
    baseline?: BaselineConfig
  }>
  sampleCount?: number
}): AuthorCurvesResponse {
  const { datasets, sampleCount = 300 } = params
  const authorRatios = new Map<string, number[]>()
  const samplePoints: number[] = []
  for (let i = 0; i < sampleCount; i++) samplePoints.push(i / (sampleCount - 1))

  for (const dataset of datasets) {
    const authorsBySolution = new Map<string, string>()
    for (const solution of dataset.solutions) {
      authorsBySolution.set(solution.name, solution.author)
    }

    const { groupRatios } = computeSolutionGroupRatios(dataset)
    for (const ratios of groupRatios.values()) {
      for (const [solutionName, ratio] of ratios.entries()) {
        const author = authorsBySolution.get(solutionName)
        if (!author) continue
        if (!authorRatios.has(author)) authorRatios.set(author, [])
        authorRatios.get(author)!.push(ratio)
      }
    }
  }

  const curves: Record<string, CurvePoint[]> = {}
  const comparisonCounts: Record<string, number> = {}
  let totalComparisons = 0

  for (const [author, ratios] of authorRatios.entries()) {
    ratios.sort((a, b) => a - b)
    comparisonCounts[author] = ratios.length
    totalComparisons += ratios.length

    const points: CurvePoint[] = []
    for (const p of samplePoints) {
      let lo = 0
      let hi = ratios.length
      while (lo < hi) {
        const mid = (lo + hi) >>> 1
        if (ratios[mid] < p) lo = mid + 1
        else hi = mid
      }
      const count = ratios.length - lo
      const percent = ratios.length > 0 ? (count / ratios.length) * 100 : 0
      points.push({ p, percent })
    }
    curves[author] = points
  }

  return { curves, comparisonCounts, totalComparisons }
}

export function computeAuthorCorrectnessSummary(params: {
  datasets: Array<{
    traces: Trace[]
    solutions: Solution[]
  }>
}): AuthorCorrectnessResponse {
  const { datasets } = params
  const aggregate = new Map<string, CorrectnessSummary>()
  const totals: CorrectnessSummary = { total: 0, passed: 0, incorrect: 0, runtime_error: 0, other: 0 }

  for (const dataset of datasets) {
    const summary = computeCorrectnessSummaryForSolutions(dataset.traces, dataset.solutions)
    for (const solution of dataset.solutions) {
      const stats = summary[solution.name]
      if (!stats) continue
      const author = solution.author || "unknown"
      const record = aggregate.get(author) ?? { total: 0, passed: 0, incorrect: 0, runtime_error: 0, other: 0 }
      record.total += stats.total
      record.passed += stats.passed
      record.incorrect += stats.incorrect
      record.runtime_error += stats.runtime_error
      record.other += stats.other
      aggregate.set(author, record)

      totals.total += stats.total
      totals.passed += stats.passed
      totals.incorrect += stats.incorrect
      totals.runtime_error += stats.runtime_error
      totals.other += stats.other
    }
  }

  const stats = Array.from(aggregate.entries()).map(([author, counts]) => ({
    author,
    ...counts,
  }))

  return { stats, totals }
}

export type SolutionTraceComparison = {
  workloadId: WorkloadGroupId
  traces: Trace[]
  baseline?: Trace
  candidate?: Trace
  baselineLatency?: number | null
  candidateLatency?: number | null
  ratio?: number | null
}

export type SolutionTraceBuckets = {
  faster: SolutionTraceComparison[]
  slower: SolutionTraceComparison[]
  incorrect: SolutionTraceComparison[]
}

export function computeBaselineTraceComparisons(params: {
  traces: Trace[]
  solutionName: string
}): SolutionTraceComparison[] {
  const { traces, solutionName } = params
  const groups = buildWorkloadGroups(traces)
  const comparisons: SolutionTraceComparison[] = []

  for (const [workloadId, groupTraces] of groups) {
    const baselineTrace = groupTraces.find((trace) => trace.solution === solutionName)
    if (!baselineTrace) continue

    const baselineLatency = baselineTrace.evaluation?.performance?.latency_ms ?? null
    comparisons.push({
      workloadId,
      traces: groupTraces,
      baseline: baselineTrace,
      candidate: baselineTrace,
      baselineLatency,
      candidateLatency: baselineLatency,
      ratio: baselineLatency && baselineLatency > 0 ? 1 : null,
    })
  }

  return comparisons
}

export function computeSolutionTraceBuckets(params: {
  traces: Trace[]
  solutions: Solution[]
  solutionName: string
  p: number
  baseline?: BaselineConfig
}): SolutionTraceBuckets {
  const { traces, solutions, solutionName, p, baseline } = params
  const solMap = solutionMap(solutions)
  const groups = buildWorkloadGroups(traces)
  const faster: SolutionTraceComparison[] = []
  const slower: SolutionTraceComparison[] = []
  const incorrect: SolutionTraceComparison[] = []

  for (const [workloadId, groupTraces] of groups) {
    const candidate = groupTraces.find((trace) => trace.solution === solutionName)
    if (!candidate) continue

    const device = deviceForTrace(candidate) || "unknown"
    const baselineName = baselineForDevice(device, baseline)
    if (!baselineName) continue
    const baselineTrace = groupTraces.find((trace) => trace.solution === baselineName)
    const baselineLatency = baselineTrace?.evaluation?.performance?.latency_ms ?? null

    const comparison: SolutionTraceComparison = {
      workloadId,
      traces: groupTraces,
      candidate,
      candidateLatency: candidate.evaluation?.performance?.latency_ms ?? null,
      baseline: baselineTrace,
      baselineLatency,
    }

    const status = candidate.evaluation?.status
    if (!status || status !== "PASSED") {
      incorrect.push(comparison)
      continue
    }

    if (baselineLatency == null || !comparison.candidateLatency) {
      slower.push({ ...comparison, ratio: null })
      continue
    }

    const ratio = baselineLatency / comparison.candidateLatency
    const target = ratio >= p ? faster : slower
    target.push({
      ...comparison,
      ratio,
    })
  }

  return { faster, slower, incorrect }
}
