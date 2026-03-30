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
  totalWorkloads: number
  coverage: Record<string, CoverageStats>
}

export type SpeedupMetrics = {
  avgSpeedup: number
  definitions: number
  workloads: number
  successRate: number
  winRate: number
  bestSolutions: Record<string, string> | null
}

export type AuthorRankingEntry = SpeedupMetrics & { author: string }

export type AuthorRankingsResponse = {
  rankings: AuthorRankingEntry[]
}

export type AuthorCorrectnessResponse = {
  stats: Array<CorrectnessSummary & { author: string }>
  totals: CorrectnessSummary
}

export type CoverageStats = {
  attempted: number
  total: number
  percent: number
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
  maxRatio: number
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
  let maxRatio = 1

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

    // Track all candidate outcomes for this workload; failures contribute ratio 0
    const ratioMap: Map<string, number> = new Map()

    for (const trace of groupTraces) {
      const solutionName = trace.solution
      if (!solutionName) continue
      if (!solMap.has(solutionName)) continue
      if (solutionName === baselineName) continue

      const status = trace.evaluation?.status
      const candidateLatency = trace.evaluation?.performance?.latency_ms
      const passed = status === "PASSED" && typeof candidateLatency === "number" && candidateLatency > 0
      if (passed) {
        const ratio = baselineLatency / candidateLatency
        maxRatio = Math.max(maxRatio, ratio)
        ratioMap.set(solutionName, ratio)
      } else if (!ratioMap.has(solutionName)) {
        ratioMap.set(solutionName, 0)
      }
    }

    groupRatios.set(workloadId, ratioMap)
  }

  return {
    groupRatios,
    nWorkloads: groupRatios.size,
    baselineNames,
    maxRatio,
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
  const { groupRatios, nWorkloads, baselineNames, maxRatio } = computeSolutionGroupRatios({ traces, solutions, baseline })
  const sampleMax = Math.max(1, maxRatio)
  const adjustedCount = Math.min(5000, Math.max(sampleCount, Math.ceil(sampleCount * sampleMax)))
  const samplePoints: number[] = []
  if (adjustedCount <= 1) {
    samplePoints.push(sampleMax)
  } else {
    const step = sampleMax / (adjustedCount - 1)
    for (let i = 0; i < adjustedCount; i++) samplePoints.push(step * i)
  }

  const curves: Record<string, CurvePoint[]> = {}
  for (const [solutionName] of solMap) {
    if (baselineNames.has(solutionName)) continue
    const ratios: number[] = []
    for (const ratiosForGroup of groupRatios.values()) {
      const value = ratiosForGroup.get(solutionName)
      if (value == null) continue
      ratios.push(value)
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
      const total = ratios.length
      const percent = total > 0 ? (count / total) * 100 : 0
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
  const authorAttempts = new Map<string, number>()
  const authorTotals = new Map<string, number>()
  let overallMaxRatio = 1
  const samplePoints: number[] = []
  let totalWorkloads = 0

  for (const dataset of datasets) {
    const { groupRatios, maxRatio, baselineNames } = computeSolutionGroupRatios(dataset)
    overallMaxRatio = Math.max(overallMaxRatio, maxRatio)
    totalWorkloads += groupRatios.size
    if (groupRatios.size === 0) continue

    const candidateSolutions = dataset.solutions.filter(
      (solution) => solution.author && !baselineNames.has(solution.name)
    )

    const solutionsPerAuthor = new Map<string, number>()
    for (const solution of candidateSolutions) {
      const author = solution.author!
      solutionsPerAuthor.set(author, (solutionsPerAuthor.get(author) ?? 0) + 1)
    }

    const workloads = groupRatios.size
    for (const [author, count] of solutionsPerAuthor.entries()) {
      const total = workloads * count
      authorTotals.set(author, (authorTotals.get(author) ?? 0) + total)
    }

    for (const ratios of groupRatios.values()) {
      for (const solution of candidateSolutions) {
        const author = solution.author!
        const ratio = ratios.get(solution.name)
        if (ratio == null) continue
        if (!authorRatios.has(author)) authorRatios.set(author, [])
        authorRatios.get(author)!.push(ratio)
        authorAttempts.set(author, (authorAttempts.get(author) ?? 0) + 1)
      }
    }
  }

  const sampleMax = Math.max(1, overallMaxRatio)
  const adjustedCount = Math.min(5000, Math.max(sampleCount, Math.ceil(sampleCount * sampleMax)))
  if (adjustedCount <= 1) {
    samplePoints.push(sampleMax)
  } else {
    const step = sampleMax / (adjustedCount - 1)
    for (let i = 0; i < adjustedCount; i++) samplePoints.push(step * i)
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

  const coverage: Record<string, CoverageStats> = {}
  const authors = new Set([...authorTotals.keys(), ...authorAttempts.keys()])
  for (const author of authors) {
    const attempted = authorAttempts.get(author) ?? 0
    const total = authorTotals.get(author) ?? 0
    const percent = total > 0 ? (attempted / total) * 100 : 0
    coverage[author] = { attempted, total, percent }
  }

  return { curves, comparisonCounts, totalComparisons, totalWorkloads, coverage }
}

function collectBaselineNames(baseline?: BaselineConfig): Set<string> {
  const names = new Set<string>()
  if (baseline?.default) names.add(baseline.default)
  if (baseline?.devices) {
    for (const value of Object.values(baseline.devices)) names.add(value)
  }
  return names
}

/**
 * Score a single solution against the baseline.
 *
 * Mirrors Python TraceSet.get_solution_score:
 * - Scans ALL traces for the solution (including workloads where baseline has
 *   no data) to determine has_fail. Any non-PASSED trace → entire score = 0.
 * - Per-workload best (lowest) passing latency for both solution and baseline.
 * - Common workloads = baseline passed AND solution attempted.
 * - Speedup = baseline_latency / solution_latency per workload, averaged.
 */
export function computeSolutionScore(params: {
  traces: Trace[]
  solutionName: string
  baseline?: BaselineConfig
}): SpeedupMetrics | null {
  const { traces, solutionName, baseline } = params

  const baselineNames = collectBaselineNames(baseline)
  if (baselineNames.has(solutionName)) return null

  let hasFail = false
  const solutionBestLatency = new Map<WorkloadGroupId, number>()
  const solutionWorkloadIds = new Set<WorkloadGroupId>()

  for (const trace of traces) {
    if (trace.solution !== solutionName) continue
    if (!trace.evaluation) continue
    const workloadId = groupIdForTrace(trace)
    solutionWorkloadIds.add(workloadId)
    if (trace.evaluation.status !== "PASSED") {
      hasFail = true
      continue
    }
    const latency = trace.evaluation.performance?.latency_ms
    if (typeof latency !== "number" || latency <= 0) continue
    const existing = solutionBestLatency.get(workloadId)
    if (existing == null || latency < existing) {
      solutionBestLatency.set(workloadId, latency)
    }
  }

  const groups = buildWorkloadGroups(traces)
  const baselineBestLatency = new Map<WorkloadGroupId, number>()

  for (const [workloadId, groupTraces] of groups) {
    if (groupTraces.length === 0) continue
    const device = deviceForTrace(groupTraces[0]) || "unknown"
    const baselineName = baselineForDevice(device, baseline)
    if (!baselineName) continue

    for (const trace of groupTraces) {
      if (trace.solution !== baselineName) continue
      if (!trace.evaluation || trace.evaluation.status !== "PASSED") continue
      const latency = trace.evaluation.performance?.latency_ms
      if (typeof latency !== "number" || latency <= 0) continue
      const existing = baselineBestLatency.get(workloadId)
      if (existing == null || latency < existing) {
        baselineBestLatency.set(workloadId, latency)
      }
    }
  }

  const commonIds: WorkloadGroupId[] = []
  for (const id of baselineBestLatency.keys()) {
    if (solutionWorkloadIds.has(id)) commonIds.push(id)
  }

  if (commonIds.length === 0) return null

  const workloads = commonIds.length

  if (hasFail) {
    return {
      avgSpeedup: 0, definitions: 1, workloads,
      successRate: 0, winRate: 0, bestSolutions: null,
    }
  }

  let sumSpeedup = 0
  let wins = 0
  for (const id of commonIds) {
    const ratio = baselineBestLatency.get(id)! / solutionBestLatency.get(id)!
    sumSpeedup += ratio
    if (ratio > 1) wins++
  }

  return {
    avgSpeedup: sumSpeedup / workloads,
    definitions: 1,
    workloads,
    successRate: 1,
    winRate: wins / workloads,
    bestSolutions: null,
  }
}

/**
 * Rank all authors by average speedup.
 *
 * Mirrors Python TraceSet.rank_authors → get_author_score → get_solution_score:
 * - For each definition, score every non-baseline solution via computeSolutionScore.
 * - Per author per definition, keep only the best-scoring solution.
 * - Author's final score = simple mean across definitions (not weighted by workload count).
 */
export function computeAuthorRankings(params: {
  datasets: Array<{
    traces: Trace[]
    solutions: Solution[]
    baseline?: BaselineConfig
    definitionName?: string
  }>
}): AuthorRankingsResponse {
  const { datasets } = params

  const authorDefScores = new Map<string, Map<string, SpeedupMetrics>>()

  for (const dataset of datasets) {
    const defName = dataset.definitionName ?? "unknown"
    const baselineNames = collectBaselineNames(dataset.baseline)

    for (const solution of dataset.solutions) {
      if (!solution.author || baselineNames.has(solution.name)) continue

      const score = computeSolutionScore({
        traces: dataset.traces,
        solutionName: solution.name,
        baseline: dataset.baseline,
      })
      if (!score) continue

      const defScores = authorDefScores.get(solution.author) ?? new Map<string, SpeedupMetrics>()
      const current = defScores.get(defName)
      if (!current || score.avgSpeedup > current.avgSpeedup) {
        defScores.set(defName, {
          ...score,
          bestSolutions: { [defName]: solution.name },
        })
      }
      authorDefScores.set(solution.author, defScores)
    }
  }

  const rankings = Array.from(authorDefScores.entries())
    .map(([author, defScores]) => {
      const scores = Array.from(defScores.values())
      const n = scores.length
      const bestSolutions: Record<string, string> = {}
      for (const [def, score] of defScores.entries()) {
        if (score.bestSolutions) Object.assign(bestSolutions, score.bestSolutions)
      }
      return {
        author,
        avgSpeedup: scores.reduce((sum, s) => sum + s.avgSpeedup, 0) / n,
        definitions: n,
        workloads: scores.reduce((sum, s) => sum + s.workloads, 0),
        successRate: scores.reduce((sum, s) => sum + s.successRate, 0) / n,
        winRate: scores.reduce((sum, s) => sum + s.winRate, 0) / n,
        bestSolutions,
      }
    })
    .sort((a, b) => {
      if (b.avgSpeedup !== a.avgSpeedup) return b.avgSpeedup - a.avgSpeedup
      return a.author.localeCompare(b.author)
    })

  return { rankings }
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
