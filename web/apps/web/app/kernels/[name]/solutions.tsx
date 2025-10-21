"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { toast } from "@flashinfer-bench/ui"
import type { Definition, Solution, Trace } from "@/lib/schemas"
import { FastPCurves } from "@/components/fast-p-chart"
import { SolutionsList, type FilterChip, getSolutionElementId } from "./solutions-list"
import { useSearchParams } from "next/navigation"
import {
  computeBaselineTraceComparisons,
  computeSolutionTraceBuckets,
  type SolutionTraceBuckets,
} from "@/lib/analytics"
import type { CurvesPayload, SolutionFiltersState, CorrectnessStats } from "./solutions-types"
import baselinesData from "@/data/baselines"
import { FastPLabel } from "@/components/fast-p-label"

const DEFAULT_MAX_VISIBLE = 10
const DEFAULT_INITIAL_VISIBLE = 5
const DEFAULT_PIN = 0.95

const initialSF: SolutionFiltersState = { languages: [], authors: [], targets: [], search: "" }

function matchesSolutionFilters(solution: Solution, filters: SolutionFiltersState) {
  if (filters.languages.length && !filters.languages.includes(solution.spec.language)) return false
  if (filters.authors.length && !filters.authors.includes(solution.author)) return false
  if (filters.targets.length && !solution.spec.target_hardware.some((target) => filters.targets.includes(target))) return false
  if (filters.search.trim()) {
    const q = filters.search.trim().toLowerCase()
    const haystack = `${solution.name} ${solution.author} ${solution.spec.language} ${solution.spec.target_hardware.join(" ")}`.toLowerCase()
    if (!haystack.includes(q)) return false
  }
  return true
}

function buildScoreMap(curves: CurvesPayload | null, p: number): Record<string, number> {
  const map: Record<string, number> = {}
  if (!curves) return map

  for (const [name, points] of Object.entries(curves.curves || {})) {
    if (!points.length) {
      map[name] = 0
      continue
    }
    const minP = points[0].p
    const maxP = points[points.length - 1].p
    const target = Math.min(Math.max(p, minP), maxP)
    let lo = 0
    let hi = points.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >>> 1
      if (points[mid].p < target) lo = mid + 1
      else hi = mid
    }
    map[name] = points[lo]?.percent ?? 0
  }
  return map
}

function compareSolutions(
  a: Solution,
  b: Solution,
  correctness: Record<string, CorrectnessStats | undefined>,
  scoreMap: Record<string, number>
) {
  const statsA = correctness[a.name]
  const statsB = correctness[b.name]
  const totalA = statsA?.total ?? 0
  const totalB = statsB?.total ?? 0
  const passedA = statsA?.passed ?? 0
  const passedB = statsB?.passed ?? 0
  const allPassedA = totalA > 0 && passedA === totalA
  const allPassedB = totalB > 0 && passedB === totalB
  const scoreA = scoreMap[a.name] ?? 0
  const scoreB = scoreMap[b.name] ?? 0
  if (allPassedA && allPassedB) {
    if (scoreB !== scoreA) return scoreB - scoreA
    if (totalB !== totalA) return totalB - totalA
    return a.name.localeCompare(b.name)
  }
  if (allPassedA !== allPassedB) {
    return allPassedA ? -1 : 1
  }
  const passRateA = totalA > 0 ? passedA / totalA : 0
  const passRateB = totalB > 0 ? passedB / totalB : 0
  if (passRateB !== passRateA) return passRateB - passRateA
  if (totalB !== totalA) return totalB - totalA
  if (scoreB !== scoreA) return scoreB - scoreA
  return a.name.localeCompare(b.name)
}

export type SolutionsTracesSectionProps = {
  definition: Definition
  solutions: Solution[]
  traces: Trace[]
  precomputed: CurvesPayload
}

export function SolutionsSection({ definition, solutions, traces, precomputed }: SolutionsTracesSectionProps) {
  const searchParams = useSearchParams()

  const [sfState, setSfState] = useState<SolutionFiltersState>(initialSF)
  const [visibleSolutions, setVisibleSolutions] = useState<Set<string>>(new Set())
  const [expandedSolution, setExpandedSolution] = useState<string | null>(null)
  const [highlightedSolution, setHighlightedSolution] = useState<string | null>(null)
  const [pinnedP, setPinnedP] = useState<number | null>(DEFAULT_PIN)
  const initialSolutionsRef = useRef<string[] | null>(null)
  const initialExpandedRef = useRef<string | null>(null)
  const lastQueryRef = useRef<string>("")
  const hasInitializedVisibleRef = useRef(false)

  const axisKeyOrder = useMemo(() => {
    const axes = new Set<string>()
    for (const trace of traces) {
      Object.keys(trace.workload?.axes || {}).forEach((axis) => axes.add(axis))
    }
    return Array.from(axes).sort()
  }, [traces])

  const [colorMap] = useState(() => new Map<string, string>())
  const palette = useMemo(
    () => [
      "#4e79a7",
      "#f28e2b",
      "#e15759",
      "#76b7b2",
      "#59a14f",
      "#edc949",
      "#af7aa1",
      "#ff9da7",
      "#9c755f",
      "#bab0ab",
      "#1f77b4",
      "#ff7f0e",
      "#2ca02c",
      "#d62728",
      "#9467bd",
      "#8c564b",
      "#e377c2",
      "#7f7f7f",
      "#bcbd22",
      "#17becf",
    ],
    []
  )

  const colorFor = useCallback(
    (name: string) => {
      if (colorMap.has(name)) return colorMap.get(name) as string
      let hash = 0
      for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0
      const color = palette[hash % palette.length]
      colorMap.set(name, color)
      return color
    },
    [colorMap, palette]
  )

  type BaselineConfig = Record<string, string>
  const baselineConfig = (baselinesData as Record<string, BaselineConfig | undefined>)[definition.name] || null
  const baselineDefault = baselineConfig?.default || null
  const fallbackBaseline = useMemo(() => {
    if (baselineDefault) return baselineDefault
    if (!baselineConfig) return null
    const firstEntry = Object.entries(baselineConfig).find(([key, value]) => key && value)
    return firstEntry ? firstEntry[1] : null
  }, [baselineConfig, baselineDefault])
  const baselineSolutionName = fallbackBaseline || null
  const baselineAvailable = baselineSolutionName != null
  const baselineHasTraces = useMemo(
    () => (baselineSolutionName ? traces.some((trace) => trace.solution === baselineSolutionName) : false),
    [traces, baselineSolutionName]
  )
  const baselineReady = baselineAvailable && baselineHasTraces

  const baselineSolution = useMemo(
    () => (baselineSolutionName ? solutions.find((solution) => solution.name === baselineSolutionName) || null : null),
    [solutions, baselineSolutionName]
  )

  const candidateSolutions = useMemo(
    () => solutions.filter((solution) => solution.name !== baselineSolutionName),
    [solutions, baselineSolutionName]
  )

  const availableLanguages = useMemo(
    () => Array.from(new Set(candidateSolutions.map((solution) => solution.spec.language))).sort(),
    [candidateSolutions]
  )

  const availableAuthors = useMemo(
    () => Array.from(new Set(candidateSolutions.map((solution) => solution.author))).sort(),
    [candidateSolutions]
  )

  const availableTargets = useMemo(
    () => Array.from(new Set(candidateSolutions.flatMap((solution) => solution.spec.target_hardware))).sort(),
    [candidateSolutions]
  )

  // Read state from URL on first render
  useEffect(() => {
    const params = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : searchParams
    const languages = (params.get("languages") || "").split(",").filter(Boolean)
    const authors = (params.get("authors") || "").split(",").filter(Boolean)
    const targets = (params.get("targets") || "").split(",").filter(Boolean)
    const search = params.get("search") || ""
    setSfState({ languages, authors, targets, search })

    const initialVisible = (params.get("solutions") || "").split(",").filter(Boolean)
    const initialExpanded = params.get("focus") || null
    const pParam = params.get("p")
    if (initialVisible.length) initialSolutionsRef.current = initialVisible
    if (initialExpanded) initialExpandedRef.current = initialExpanded
    if (pParam != null) setPinnedP(Math.max(0, Number(pParam)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (hasInitializedVisibleRef.current) return
    if (!baselineAvailable) return
    const scoreLookup = buildScoreMap(precomputed as CurvesPayload, DEFAULT_PIN)
    const ranked = candidateSolutions
      .slice()
      .sort((a, b) =>
        compareSolutions(
          a,
          b,
          precomputed.correctness || {},
          scoreLookup
        )
      )

    const desiredCount = Math.min(DEFAULT_MAX_VISIBLE, Math.min(DEFAULT_INITIAL_VISIBLE, ranked.length || 0))
    const fromUrl = initialSolutionsRef.current
    const selected = new Set<string>()

    if (fromUrl && fromUrl.length) {
      fromUrl.forEach((name) => {
        if (ranked.some((solution) => solution.name === name)) {
          selected.add(name)
        }
      })
    }

    for (const solution of ranked) {
      if (selected.size >= desiredCount) break
      selected.add(solution.name)
    }

    if (!selected.size && ranked.length) {
      selected.add(ranked[0].name)
    }

    const filteredSelection = new Set(Array.from(selected).filter((name) => name !== baselineSolutionName))
    setVisibleSolutions(filteredSelection)
    if (initialExpandedRef.current) setExpandedSolution(initialExpandedRef.current)
    initialSolutionsRef.current = null
    initialExpandedRef.current = null
    hasInitializedVisibleRef.current = true
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baselineAvailable, baselineSolutionName, candidateSolutions, precomputed])

  // Keep URL in sync with state
  useEffect(() => {
    if (typeof window === "undefined") return
    const params = new URLSearchParams(window.location.search)

    params.delete("languages")
    params.delete("authors")
    params.delete("targets")
    params.delete("search")
    params.delete("solutions")
    params.delete("focus")
    params.delete("p")

    if (sfState.languages.length) params.set("languages", sfState.languages.join(","))
    if (sfState.authors.length) params.set("authors", sfState.authors.join(","))
    if (sfState.targets.length) params.set("targets", sfState.targets.join(","))
    if (sfState.search) params.set("search", sfState.search)

    const selectedSolutions = Array.from(visibleSolutions)
    if (selectedSolutions.length) params.set("solutions", selectedSolutions.join(","))
    if (expandedSolution) params.set("focus", expandedSolution)
    if (pinnedP != null) params.set("p", pinnedP.toFixed(2))

    const next = params.toString()
    if (next !== lastQueryRef.current) {
      lastQueryRef.current = next
      const newUrl = `${window.location.pathname}${next ? `?${next}` : ""}`
      window.history.replaceState(null, "", newUrl)
    }
  }, [sfState, visibleSolutions, expandedSolution, pinnedP])

  const filteredSolutions = useMemo(
    () => candidateSolutions.filter((solution) => matchesSolutionFilters(solution, sfState)),
    [candidateSolutions, sfState]
  )

  const filteredCurves: CurvesPayload | null = useMemo(() => {
    const allowed = new Set(filteredSolutions.map((s) => s.name))
    const curves = Object.fromEntries(
      Object.entries(precomputed.curves).filter(([name]) => allowed.has(name))
    )
    const correctness = Object.fromEntries(
      Object.entries(precomputed.correctness).filter(([name]) => allowed.has(name))
    )
    return { curves, correctness, nWorkloads: precomputed.nWorkloads }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [precomputed, filteredSolutions])

  // Remove selections that are no longer visible
  useEffect(() => {
    setVisibleSolutions((current) => {
      const allowed = new Set(filteredSolutions.map((s) => s.name))
      const next = new Set(Array.from(current).filter((name) => allowed.has(name)))
      if (next.size === current.size) return current
      return next
    })
  }, [filteredSolutions])

  useEffect(() => {
    if (!highlightedSolution) return
    if (!filteredSolutions.some((solution) => solution.name === highlightedSolution)) {
      setHighlightedSolution(null)
    }
  }, [filteredSolutions, highlightedSolution])

  useEffect(() => {
    if (!highlightedSolution) return
    setVisibleSolutions((current) => {
      if (current.has(highlightedSolution)) return current
      const next = new Set(current)
      next.add(highlightedSolution)
      return next
    })
  }, [highlightedSolution])

  useEffect(() => {
    if (expandedSolution && expandedSolution !== baselineSolutionName && !filteredSolutions.some((s) => s.name === expandedSolution)) {
      setExpandedSolution(null)
    }
  }, [filteredSolutions, expandedSolution, baselineSolutionName])

  const scoreMap = useMemo(() => buildScoreMap(filteredCurves, pinnedP ?? DEFAULT_PIN), [filteredCurves, pinnedP])

  const sortedSolutions = useMemo(() => {
    const correctness = filteredCurves?.correctness || {}
    return filteredSolutions.slice().sort((a, b) => compareSolutions(a, b, correctness, scoreMap))
  }, [filteredSolutions, filteredCurves?.correctness, scoreMap])

  const displaySolutions = useMemo(
    () => (baselineSolution ? [baselineSolution, ...sortedSolutions] : sortedSolutions),
    [baselineSolution, sortedSolutions]
  )

  const traceBuckets: SolutionTraceBuckets | null = useMemo(() => {
    if (!expandedSolution || pinnedP == null || !baselineAvailable) return null
    return computeSolutionTraceBuckets({
      traces,
      solutions,
      solutionName: expandedSolution,
      p: pinnedP,
      baseline: baselineSolutionName ? { default: baselineSolutionName } : undefined,
    })
  }, [expandedSolution, pinnedP, traces, solutions, baselineAvailable, baselineSolutionName])

  const baselineComparisons = useMemo(() => {
    if (!expandedSolution || expandedSolution !== baselineSolutionName) return null
    return computeBaselineTraceComparisons({ traces, solutionName: expandedSolution })
  }, [expandedSolution, baselineSolutionName, traces])

  const filterChips: FilterChip[] = useMemo(() => {
    const chips: FilterChip[] = []
    for (const lang of sfState.languages) {
      chips.push({
        label: `Lang:${lang}`,
        onRemove: () => setSfState((state) => ({ ...state, languages: state.languages.filter((l) => l !== lang) })),
      })
    }
    for (const author of sfState.authors) {
      chips.push({
        label: `Author:${author}`,
        onRemove: () => setSfState((state) => ({ ...state, authors: state.authors.filter((a) => a !== author) })),
      })
    }
    for (const target of sfState.targets) {
      chips.push({
        label: `Target:${target}`,
        onRemove: () => setSfState((state) => ({ ...state, targets: state.targets.filter((t) => t !== target) })),
      })
    }
    if (sfState.search) {
      chips.push({ label: `Search:${sfState.search}`, onRemove: () => setSfState((state) => ({ ...state, search: "" })) })
    }
    return chips
  }, [sfState])

  const handleToggleSolution = useCallback(
    (name: string) => {
      setVisibleSolutions((current) => {
        const next = new Set(current)
        if (next.has(name)) {
          next.delete(name)
          return next
        }
        if (next.size >= DEFAULT_MAX_VISIBLE) {
          toast({ title: "Too many lines", description: `Limit ${DEFAULT_MAX_VISIBLE} curves for clarity.`, variant: "destructive" })
          return current
        }
        colorFor(name)
        next.add(name)
        return next
      })
    },
    [colorFor]
  )

  const handleExpandSolution = useCallback((name: string) => {
    setExpandedSolution((current) => (current === name ? null : name))
  }, [])

  const handlePinDefault = useCallback(() => {
    setPinnedP(DEFAULT_PIN)
  }, [])

  const handleOpenTrace = useCallback((trace: Trace) => {
    if (typeof window === "undefined") return
    const parts = [trace.definition || "trace", trace.solution || "workload", trace.workload?.uuid || "unknown"]
    const traceId = parts.join("-").replace(/[^a-zA-Z0-9-_]/g, "_")
    try {
      window.sessionStorage.setItem(`trace-${traceId}`, JSON.stringify(trace))
    } catch (error) {
      console.error("failed to persist trace for viewer", error)
      toast({ title: "Trace viewer", description: "Unable to open trace viewer.", variant: "destructive" })
      return
    }
    window.open(`/viewer?trace=${encodeURIComponent(traceId)}`, "_blank")
  }, [])

  const counts = useMemo(
    () => ({
      solutions: Object.keys(filteredCurves?.curves || {}).length,
      workloads: filteredCurves?.nWorkloads || 0,
    }),
    [filteredCurves]
  )

  const focusSolution = useCallback((name: string) => {
    const exists = displaySolutions.some((solution) => solution.name === name)
    if (!exists) return
    setExpandedSolution(name)
    if (typeof window !== "undefined") {
      requestAnimationFrame(() => {
        const element = document.getElementById(getSolutionElementId(name))
        if (!element) return
        const headerOffset = 122
        const rect = element.getBoundingClientRect()
        const targetY = rect.top + window.scrollY - headerOffset
        window.scrollTo({ top: targetY, behavior: "smooth" })
      })
    }
  }, [displaySolutions])

  const inspectHighlightedSolution = useCallback(() => {
    if (!highlightedSolution) return
    focusSolution(highlightedSolution)
  }, [focusSolution, highlightedSolution])

  const handleHoverP = useCallback((value: number | null) => {
    void value
  }, [])

  return (
    <section id="solutions" className="space-y-4">
      <div>
        <h2 className="text-2xl font-semibold">Results</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Top-5 solutions at <FastPLabel className="font-medium" value={0.95} /> are displayed by default.
        </p>
      </div>

      <FastPCurves
        curves={filteredCurves?.curves || {}}
        visible={visibleSolutions}
        onHoverP={handleHoverP}
        onPinP={setPinnedP}
        pinnedP={pinnedP}
        baselineLabel={baselineDefault || fallbackBaseline || "Not specified"}
        comparisonCount={counts.workloads}
        baselineAvailable={baselineReady}
        colorFor={colorFor}
        scoreboard={[]}
        highlighted={highlightedSolution}
        onHighlightChange={setHighlightedSolution}
        highlightContext="list"
        onInspectHighlighted={inspectHighlightedSolution}
        correctness={filteredCurves?.correctness || {}}
      />

      <SolutionsList
        solutions={displaySolutions}
        visibleSolutions={visibleSolutions}
        onToggleSolution={handleToggleSolution}
        onExpandSolution={handleExpandSolution}
        expandedSolution={expandedSolution}
        correctness={filteredCurves?.correctness || {}}
        colorFor={colorFor}
        pinnedP={pinnedP}
        onPinDefault={handlePinDefault}
        traceBuckets={traceBuckets}
        axisKeyOrder={axisKeyOrder}
        filterChips={filterChips}
        onOpenTrace={handleOpenTrace}
        stats={counts}
        filters={sfState}
        onSearchChange={(value) => setSfState((state) => ({ ...state, search: value }))}
        onToggleLanguage={(language, checked) =>
          setSfState((state) => ({
            ...state,
            languages: checked
              ? [...state.languages, language]
              : state.languages.filter((item) => item !== language),
          }))
        }
        onToggleAuthor={(author, checked) =>
          setSfState((state) => ({
            ...state,
            authors: checked
              ? [...state.authors, author]
              : state.authors.filter((item) => item !== author),
          }))
        }
        onToggleTarget={(target, checked) =>
          setSfState((state) => ({
            ...state,
            targets: checked
              ? [...state.targets, target]
              : state.targets.filter((item) => item !== target),
          }))
        }
        onResetFilters={() => setSfState(initialSF)}
        availableLanguages={availableLanguages}
        availableAuthors={availableAuthors}
        availableTargets={availableTargets}
        baselineSolutionName={baselineSolutionName}
        baselineComparisons={baselineComparisons}
      />
    </section>
  )
}
