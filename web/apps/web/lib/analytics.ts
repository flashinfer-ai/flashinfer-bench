import { Trace, Solution } from "./schemas"

export type WorkloadGroupId = string

export type WorkloadFilters = {
  // axis name -> [min, max] inclusive range filter
  axisRanges?: Record<string, { min: number; max: number }>
  // acceptable hardware/device identifiers
  devices?: string[]
  // when true, only consider PASSED evaluations when forming groups/denominator
  onlyPassed?: boolean
}

export type SolutionFilters = {
  languages?: string[]
  authors?: string[]
  targets?: string[]
  search?: string
}

export type CurvePoint = { p: number; percent: number }

export type CurvesResponse = {
  nWorkloads: number
  curves: Record<string, CurvePoint[]>
  // simple correctness summary per solution (counts)
  correctness: Record<
    string,
    {
      total: number
      passed: number
      incorrect: number
      runtime_error: number
      other: number
    }
  >
}

type Grouped = Map<WorkloadGroupId, Trace[]>

function stableStringify(obj: any): string {
  if (obj === null || typeof obj !== "object") return JSON.stringify(obj)
  if (Array.isArray(obj)) return `[${obj.map(stableStringify).join(",")}]`
  return `{${Object.keys(obj)
    .sort()
    .map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`)
    .join(",")}}`
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

function passesFilters(t: Trace, wf?: WorkloadFilters): boolean {
  if (!wf) return true
  const { axisRanges, devices, onlyPassed } = wf
  if (axisRanges) {
    for (const [axis, range] of Object.entries(axisRanges)) {
      const val = (t.workload?.axes as any)?.[axis]
      if (typeof val !== "number") return false
      if (val < range.min || val > range.max) return false
    }
  }
  if (devices && devices.length) {
    const dev = t.evaluation?.environment?.device || t.evaluation?.environment?.hardware
    if (!dev || !devices.includes(dev)) return false
  }
  if (onlyPassed) {
    if (t.evaluation?.status !== "PASSED") return false
  }
  return true
}

function passesSolutionFilters(s: Solution, sf?: SolutionFilters): boolean {
  if (!sf) return true
  const { languages, authors, targets, search } = sf
  if (languages && languages.length) {
    if (!languages.map((v) => v.toLowerCase()).includes(s.spec.language.toLowerCase())) return false
  }
  if (authors && authors.length) {
    if (!authors.map((v) => v.toLowerCase()).includes(s.author.toLowerCase())) return false
  }
  if (targets && targets.length) {
    if (!s.spec.target_hardware.some((t) => targets.includes(t))) return false
  }
  if (search && search.trim()) {
    const q = search.trim().toLowerCase()
    const hay = `${s.name} ${s.description} ${s.author} ${s.spec.language} ${s.spec.target_hardware.join(" ")}`.toLowerCase()
    if (!hay.includes(q)) return false
  }
  return true
}

function isIncorrectStatus(status?: string | null): boolean {
  return status === "INCORRECT_SHAPE" || status === "INCORRECT_NUMERICAL" || status === "INCORRECT_DTYPE"
}

function isRuntimeError(status?: string | null): boolean {
  return status === "RUNTIME_ERROR" || status === "COMPILE_ERROR" || status === "TIMEOUT"
}

export function buildWorkloadGroups(traces: Trace[], wf?: WorkloadFilters): Grouped {
  const m: Grouped = new Map()
  for (const t of traces) {
    if (!t || !t.evaluation) continue
    if (!passesFilters(t, wf)) continue
    const id = groupIdForTrace(t)
    if (!m.has(id)) m.set(id, [])
    m.get(id)!.push(t)
  }
  return m
}

export function solutionMap(solutions: Solution[], sf?: SolutionFilters): Map<string, Solution> {
  const map = new Map<string, Solution>()
  for (const s of solutions) {
    if (passesSolutionFilters(s, sf)) {
      map.set(s.name, s)
    }
  }
  return map
}

export type BaselineConfig = {
  default?: string
  devices?: Record<string, string>
}

export function computeCorrectnessSummary(traces: Trace[], solutions: Solution[], wf?: WorkloadFilters, sf?: SolutionFilters) {
  const solMap = solutionMap(solutions, sf)
  const summary: CurvesResponse["correctness"] = {}
  for (const s of solMap.values()) {
    summary[s.name] = { total: 0, passed: 0, incorrect: 0, runtime_error: 0, other: 0 }
  }
  for (const t of traces) {
    if (!t.solution || !solMap.has(t.solution)) continue
    // Ignore onlyPassed when computing correctness; still honor devices/axisRanges
    const wfNoOnly: WorkloadFilters | undefined = wf
      ? { ...wf, onlyPassed: false }
      : undefined
    if (!passesFilters(t, wfNoOnly)) continue
    const status = t.evaluation?.status || null
    const rec = summary[t.solution] || (summary[t.solution] = { total: 0, passed: 0, incorrect: 0, runtime_error: 0, other: 0 })
    rec.total += 1
    if (status === "PASSED") rec.passed += 1
    else if (isIncorrectStatus(status || undefined)) rec.incorrect += 1
    else if (isRuntimeError(status || undefined)) rec.runtime_error += 1
    else rec.other += 1
  }
  return summary
}

export function computeWinAtPCurves(params: {
  traces: Trace[]
  solutions: Solution[]
  workloadFilters?: WorkloadFilters
  solutionFilters?: SolutionFilters
  sampleCount?: number // e.g. 201 points from 0..1
  baseline?: BaselineConfig
}): CurvesResponse {
  const { traces, solutions, workloadFilters, solutionFilters, sampleCount = 201, baseline } = params
  const solMap = solutionMap(solutions, solutionFilters)
  const groups = buildWorkloadGroups(traces, workloadFilters)

  // For each group, compute baseline and per-solution ratios r = Lb/La
  const groupRatios: Map<WorkloadGroupId, Map<string, number>> = new Map()
  for (const [gid, trs] of groups) {
    if (trs.length === 0) continue
    const device = deviceForTrace(trs[0]) || "unknown"
    const baselineName = (baseline?.devices && baseline.devices[device]) || baseline?.default || null
    if (!baselineName) continue
    const baselineTrace = trs.find((t) => t.solution === baselineName)
    const baselineLatency = baselineTrace?.evaluation?.performance?.latency_ms
    if (typeof baselineLatency !== "number" || baselineLatency <= 0) continue
    const ratios = new Map<string, number>()
    for (const t of trs) {
      if (!t.solution) continue
      if (!solMap.has(t.solution)) continue
      if (t.solution === baselineName) continue
      const lat = t.evaluation?.performance?.latency_ms
      if (typeof lat !== "number" || lat <= 0) continue
      ratios.set(t.solution, baselineLatency / lat)
    }
    if (ratios.size > 0) {
      groupRatios.set(gid, ratios)
    }
  }

  const nWorkloads = groupRatios.size

  // Prepare evenly spaced p in [0,1]
  const points: number[] = []
  for (let i = 0; i < sampleCount; i++) points.push(i / (sampleCount - 1))

  // Precompute per-solution array of ratios, sorted for binary search
  const perSolutionSorted: Record<string, number[]> = {}
  for (const s of solMap.values()) {
    const ratios: number[] = []
    for (const [, m] of groupRatios) {
      const r = m.get(s.name)
      ratios.push(typeof r === "number" ? r : 0)
    }
    ratios.sort((a, b) => a - b)
    perSolutionSorted[s.name] = ratios
  }

  const baselineNames = new Set<string>()
  if (baseline?.default) baselineNames.add(baseline.default)
  if (baseline?.devices) {
    for (const value of Object.values(baseline.devices)) {
      baselineNames.add(value)
    }
  }

  const curves: Record<string, CurvePoint[]> = {}
  for (const [sname, ratios] of Object.entries(perSolutionSorted)) {
    if (baselineNames.has(sname)) continue
    const pts: CurvePoint[] = []
    for (const p of points) {
      // fraction of groups where r >= p (right-continuous)
      // lower_bound for p in sorted ratios
      let lo = 0, hi = ratios.length
      while (lo < hi) {
        const mid = (lo + hi) >>> 1
        if (ratios[mid] < p) lo = mid + 1
        else hi = mid
      }
      const cnt = ratios.length - lo
      const percent = nWorkloads > 0 ? (cnt / nWorkloads) * 100 : 0
      pts.push({ p, percent })
    }
    curves[sname] = pts
  }

  const correctness = computeCorrectnessSummary(traces, solutions, workloadFilters, solutionFilters)
  return { nWorkloads, curves, correctness }
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
    const baselineName = (baseline?.devices && baseline.devices[device]) || baseline?.default || null
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
