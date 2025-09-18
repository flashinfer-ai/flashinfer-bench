"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { toast } from "@flashinfer-bench/ui"
import type { Definition, Solution, Trace } from "@/lib/schemas"
import { WinAtPCurves, type ScoreboardEntry } from "./win-at-p"
import { SolutionsList, type FilterChip } from "./solutions-list"
import { useSearchParams } from "next/navigation"
import { computeSolutionTraceBuckets, type SolutionTraceBuckets } from "@/lib/analytics"
import type { CurvesPayload, SolutionFiltersState, CorrectnessStats } from "./solutions-types"

const DEFAULT_MAX_VISIBLE = 10
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
    const index = Math.round(Math.max(0, Math.min(1, p)) * (points.length - 1))
    map[name] = points[index]?.percent ?? 0
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
}

export function SolutionsSection({ definition, solutions, traces }: SolutionsTracesSectionProps) {
  const searchParams = useSearchParams()

  const [sfState, setSfState] = useState<SolutionFiltersState>(initialSF)
  const [curves, setCurves] = useState<CurvesPayload | null>(null)
  const [visibleSolutions, setVisibleSolutions] = useState<Set<string>>(new Set())
  const [expandedSolution, setExpandedSolution] = useState<string | null>(null)
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

  const availableLanguages = useMemo(
    () => Array.from(new Set(solutions.map((solution) => solution.spec.language))).sort(),
    [solutions]
  )

  const availableAuthors = useMemo(
    () => Array.from(new Set(solutions.map((solution) => solution.author))).sort(),
    [solutions]
  )

  const availableTargets = useMemo(
    () => Array.from(new Set(solutions.flatMap((solution) => solution.spec.target_hardware))).sort(),
    [solutions]
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
    if (pParam != null) setPinnedP(Math.max(0, Math.min(1, Number(pParam))))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Fetch curves when filters change
  useEffect(() => {
    const params = new URLSearchParams()
    if (sfState.languages.length) params.set("languages", sfState.languages.join(","))
    if (sfState.authors.length) params.set("authors", sfState.authors.join(","))
    if (sfState.targets.length) params.set("targets", sfState.targets.join(","))
    if (sfState.search) params.set("search", sfState.search)

    fetch(`/api/definitions/${encodeURIComponent(definition.name)}/curves?${params.toString()}`)
      .then((response) => response.json())
      .then((data: CurvesPayload) => {
        setCurves(data)

        if (!hasInitializedVisibleRef.current) {
          const scoreLookup = buildScoreMap(data, DEFAULT_PIN)
          const ranked = solutions
            .slice()
            .sort((a, b) =>
              compareSolutions(
                a,
                b,
                data.correctness || {},
                scoreLookup
              )
            )

          const allPassedCount = ranked.filter((solution) => {
            const stats = data.correctness?.[solution.name]
            return stats && stats.total > 0 && stats.passed === stats.total
          }).length

          const desiredCount = allPassedCount > 0
            ? Math.min(DEFAULT_MAX_VISIBLE, Math.min(4, allPassedCount))
            : Math.min(DEFAULT_MAX_VISIBLE, 4)
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

          setVisibleSolutions(selected)
          if (initialExpandedRef.current) setExpandedSolution(initialExpandedRef.current)
          initialSolutionsRef.current = null
          initialExpandedRef.current = null
          hasInitializedVisibleRef.current = true
        }
      })
      .catch((error) => console.error("failed to fetch curves", error))
  }, [definition.name, sfState, solutions])

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
    () => solutions.filter((solution) => matchesSolutionFilters(solution, sfState)),
    [solutions, sfState]
  )

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
    if (expandedSolution && !filteredSolutions.some((s) => s.name === expandedSolution)) {
      setExpandedSolution(null)
    }
  }, [filteredSolutions, expandedSolution])

  const scoreMap = useMemo(() => buildScoreMap(curves, pinnedP ?? DEFAULT_PIN), [curves, pinnedP])

  const scoreboard: ScoreboardEntry[] = useMemo(() => {
    if (!visibleSolutions.size) return []
    return Array.from(visibleSolutions)
      .map((name) => ({ name, percent: scoreMap[name] ?? 0 }))
      .sort((a, b) => b.percent - a.percent)
  }, [visibleSolutions, scoreMap])

  const sortedSolutions = useMemo(() => {
    const correctness = curves?.correctness || {}
    return filteredSolutions.slice().sort((a, b) => compareSolutions(a, b, correctness, scoreMap))
  }, [filteredSolutions, curves?.correctness, scoreMap])

  const traceBuckets: SolutionTraceBuckets | null = useMemo(() => {
    if (!expandedSolution || pinnedP == null) return null
    return computeSolutionTraceBuckets({
      traces,
      solutions,
      solutionName: expandedSolution,
      p: pinnedP,
    })
  }, [expandedSolution, pinnedP, traces, solutions])

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
    toast({ title: "Trace viewer", description: `Trace ${trace.workload?.uuid || trace.solution} coming soon.` })
  }, [])

  const counts = useMemo(
    () => ({
      solutions: Object.keys(curves?.curves || {}).length,
      workloads: curves?.nWorkloads || 0,
    }),
    [curves]
  )

  return (
    <section id="solutions" className="space-y-6">
      <h2 className="text-2xl font-semibold">Results</h2>

      <WinAtPCurves
        curves={curves?.curves || {}}
        visible={visibleSolutions}
        onHoverP={() => {}}
        onPinP={setPinnedP}
        pinnedP={pinnedP}
        headline={`n = ${counts.workloads} workloads. Baseline prefers FlashInfer else fastest.`}
        colorFor={colorFor}
        scoreboard={scoreboard}
      />

      <SolutionsList
        solutions={sortedSolutions}
        visibleSolutions={visibleSolutions}
        onToggleSolution={handleToggleSolution}
        onExpandSolution={handleExpandSolution}
        expandedSolution={expandedSolution}
        correctness={curves?.correctness || {}}
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
      />
    </section>
  )
}
