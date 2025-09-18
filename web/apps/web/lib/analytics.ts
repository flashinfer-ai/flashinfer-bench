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

export function pickBaselineLatency(traces: Trace[], solMap: Map<string, Solution>): number | null {
  // Prefer baseline authored by flashinfer
  const preferred = traces
    .filter((t) => !!t.solution && !!t.evaluation?.performance?.latency_ms)
    .filter((t) => {
      const s = t.solution ? solMap.get(t.solution) : undefined
      return s && s.author?.toLowerCase() === "flashinfer"
    })
    .sort((a, b) => (a.evaluation!.performance!.latency_ms! - b.evaluation!.performance!.latency_ms!))
  if (preferred.length > 0) return preferred[0].evaluation!.performance!.latency_ms!

  // Fallback: fastest run in group
  const all = traces
    .filter((t) => !!t.evaluation?.performance?.latency_ms)
    .sort((a, b) => (a.evaluation!.performance!.latency_ms! - b.evaluation!.performance!.latency_ms!))
  return all.length > 0 ? all[0].evaluation!.performance!.latency_ms! : null
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
}): CurvesResponse {
  const { traces, solutions, workloadFilters, solutionFilters, sampleCount = 201 } = params
  const solMap = solutionMap(solutions, solutionFilters)
  const groups = buildWorkloadGroups(traces, workloadFilters)

  // For each group, compute baseline and per-solution ratios r = Lb/La
  const groupRatios: Map<WorkloadGroupId, Map<string, number>> = new Map()
  for (const [gid, trs] of groups) {
    const baselineLatency = pickBaselineLatency(trs, solMap)
    if (baselineLatency == null) continue
    const ratios = new Map<string, number>()
    for (const t of trs) {
      if (!t.solution) continue
      if (!solMap.has(t.solution)) continue
      const lat = t.evaluation?.performance?.latency_ms
      if (typeof lat !== "number" || lat <= 0) continue
      ratios.set(t.solution, baselineLatency / lat)
    }
    groupRatios.set(gid, ratios)
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

  const curves: Record<string, CurvePoint[]> = {}
  for (const [sname, ratios] of Object.entries(perSolutionSorted)) {
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
}): SolutionTraceBuckets {
  const { traces, solutions, solutionName, p } = params
  const solMap = solutionMap(solutions)
  const groups = buildWorkloadGroups(traces)
  const faster: SolutionTraceComparison[] = []
  const slower: SolutionTraceComparison[] = []
  const incorrect: SolutionTraceComparison[] = []

  for (const [workloadId, groupTraces] of groups) {
    const candidate = groupTraces.find((trace) => trace.solution === solutionName)
    if (!candidate) continue

    const comparison: SolutionTraceComparison = {
      workloadId,
      traces: groupTraces,
      candidate,
      candidateLatency: candidate.evaluation?.performance?.latency_ms ?? null,
    }

    const status = candidate.evaluation?.status
    if (!status || status !== "PASSED") {
      incorrect.push(comparison)
      continue
    }

    const baselineLatency = pickBaselineLatency(groupTraces, solMap)
    if (baselineLatency == null || !comparison.candidateLatency) {
      slower.push({ ...comparison, baselineLatency, ratio: null })
      continue
    }

    const baselineTrace = groupTraces.find(
      (trace) => trace.evaluation?.performance?.latency_ms === baselineLatency
    )

    const ratio = baselineLatency / comparison.candidateLatency
    const target = ratio >= p ? faster : slower
    target.push({
      ...comparison,
      baseline: baselineTrace,
      baselineLatency,
      ratio,
    })
  }

  return { faster, slower, incorrect }
}
