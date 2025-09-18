"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Button, toast } from "@flashinfer-bench/ui"
import type { Definition, Solution, Trace } from "@/lib/schemas"
import { SolutionsList } from "./solutions-list"
import { WinAtPCurves } from "./win-at-p-curves"
import { WorkloadsTable } from "./workloads-table"
import { SolutionsTracesToolbar } from "./solutions-traces-toolbar"
import { SideDrawer } from "./side-drawer"
import { useSearchParams } from "next/navigation"
import { X } from "lucide-react"
import type { CurvesPayload, SolutionFiltersState, WorkloadFiltersState } from "./solutions-traces-types"

const MAX_VISIBLE_SOLUTIONS = 10

const COLOR_PALETTE = [
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
]

function deriveUniqueDevices(traces: Trace[]): string[] {
  const devices = new Set<string>()
  for (const trace of traces) {
    const device = trace.evaluation?.environment.device || trace.evaluation?.environment.hardware
    if (device) devices.add(device)
  }
  return Array.from(devices).sort()
}

function deriveAxisRanges(traces: Trace[]): Record<string, { min: number; max: number }> {
  const ranges: Record<string, { min: number; max: number }> = {}
  for (const trace of traces) {
    const axes = trace.workload?.axes || {}
    for (const [axis, value] of Object.entries(axes)) {
      if (typeof value !== "number") continue
      if (!ranges[axis]) ranges[axis] = { min: value, max: value }
      else {
        ranges[axis].min = Math.min(ranges[axis].min, value)
        ranges[axis].max = Math.max(ranges[axis].max, value)
      }
    }
  }
  return ranges
}

export type SolutionsTracesSectionProps = {
  definition: Definition
  solutions: Solution[]
  traces: Trace[]
}

export function SolutionsTracesSection({ definition, solutions, traces }: SolutionsTracesSectionProps) {
  const initialWorkloadFilters: WorkloadFiltersState = { axisRanges: {}, devices: [], onlyPassed: true }
  const initialSolutionFilters: SolutionFiltersState = { languages: [], authors: [], targets: [], search: "" }

  const searchParams = useSearchParams()
  const [workloadFilters, setWorkloadFilters] = useState<WorkloadFiltersState>(initialWorkloadFilters)
  const [solutionFilters, setSolutionFilters] = useState<SolutionFiltersState>(initialSolutionFilters)
  const initialSolutionsRef = useRef<string[] | null>(null)
  const initialFocusRef = useRef<string | null>(null)

  const [curves, setCurves] = useState<CurvesPayload | null>(null)
  const [visibleSolutions, setVisibleSolutions] = useState<Set<string>>(new Set())
  const [focusedSolution, setFocusedSolution] = useState<string | null>(null)
  const [pinnedP, setPinnedP] = useState<number | null>(null)
  const [sortScores, setSortScores] = useState<Record<string, number>>({})
  const [searchQuery, setSearchQuery] = useState("")

  const [isWorkloadDrawerOpen, setWorkloadDrawerOpen] = useState(false)
  const [isSolutionDrawerOpen, setSolutionDrawerOpen] = useState(false)

  const devices = useMemo(() => deriveUniqueDevices(traces), [traces])
  const axisRangesAll = useMemo(() => deriveAxisRanges(traces), [traces])
  const axisKeyOrder = useMemo(() => Object.keys(axisRangesAll).sort(), [axisRangesAll])

  const [colorMap] = useState(() => new Map<string, string>())
  const colorFor = useCallback(
    (name: string) => {
      if (colorMap.has(name)) return colorMap.get(name) as string
      let hash = 0
      for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0
      const color = COLOR_PALETTE[hash % COLOR_PALETTE.length]
      colorMap.set(name, color)
      return color
    },
    [colorMap],
  )

  useEffect(() => {
    const onlyPassed = searchParams.get("onlyPassed") === "1"
    const devicesFromUrl = (searchParams.get("devices") || "").split(",").filter(Boolean)
    const axisRangesParam = searchParams.get("axisRanges")
    let axisRanges: WorkloadFiltersState["axisRanges"] = {}
    if (axisRangesParam) {
      try {
        axisRanges = JSON.parse(axisRangesParam)
      } catch (error) {
        toast({ title: "Invalid axisRanges", description: "Failed to parse axisRanges from URL.", variant: "destructive" })
      }
    }
    setWorkloadFilters((current) => ({ ...current, onlyPassed, devices: devicesFromUrl, axisRanges }))

    const languages = (searchParams.get("languages") || "").split(",").filter(Boolean)
    const authors = (searchParams.get("authors") || "").split(",").filter(Boolean)
    const targets = (searchParams.get("targets") || "").split(",").filter(Boolean)
    const search = searchParams.get("search") || ""
    setSolutionFilters({ languages, authors, targets, search })

    const solutionsParam = (searchParams.get("solutions") || "").split(",").filter(Boolean)
    const focusParam = searchParams.get("focus") || null
    if (solutionsParam.length) initialSolutionsRef.current = solutionsParam
    if (focusParam) initialFocusRef.current = focusParam
    const pinnedParam = searchParams.get("p")
    if (pinnedParam != null) setPinnedP(Math.max(0, Math.min(1, Number(pinnedParam))))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const params = new URLSearchParams()
    if (workloadFilters.onlyPassed) params.set("onlyPassed", "1")
    if (workloadFilters.devices.length) params.set("devices", workloadFilters.devices.join(","))
    if (Object.keys(workloadFilters.axisRanges).length) params.set("axisRanges", JSON.stringify(workloadFilters.axisRanges))
    if (solutionFilters.languages.length) params.set("languages", solutionFilters.languages.join(","))
    if (solutionFilters.authors.length) params.set("authors", solutionFilters.authors.join(","))
    if (solutionFilters.targets.length) params.set("targets", solutionFilters.targets.join(","))
    if (solutionFilters.search) params.set("search", solutionFilters.search)

    fetch(`/api/definitions/${encodeURIComponent(definition.name)}/curves?${params.toString()}`)
      .then((response) => response.json())
      .then((data: CurvesPayload) => {
        setCurves(data)
        if (visibleSolutions.size === 0) {
          if (initialSolutionsRef.current && initialSolutionsRef.current.length) {
            setVisibleSolutions(new Set(initialSolutionsRef.current))
            setFocusedSolution(initialFocusRef.current || initialSolutionsRef.current[0] || null)
            initialSolutionsRef.current = null
            initialFocusRef.current = null
          } else {
            const sortedByPass = Object.entries(data.correctness || {})
              .filter(([, stats]) => (stats?.passed || 0) > 0)
              .sort((a, b) => (b[1]?.passed || 0) - (a[1]?.passed || 0) || a[0].localeCompare(b[0]))
              .map(([name]) => name)
            const initial = sortedByPass.slice(0, MAX_VISIBLE_SOLUTIONS)
            setVisibleSolutions(new Set(initial))
            setFocusedSolution(initial[0] || null)
          }
        }
      })
      .catch((error) => console.error("failed to fetch curves", error))
  }, [definition.name, workloadFilters, solutionFilters, visibleSolutions.size])

  useEffect(() => {
    if (pinnedP == null) setPinnedP(0.95)
  }, [pinnedP])

  const lastQueryRef = useRef<string>("")
  useEffect(() => {
    const isBrowser = typeof window !== "undefined"
    if (!isBrowser) return
    const params = new URLSearchParams(window.location.search)

    params.delete("onlyPassed")
    params.delete("devices")
    params.delete("axisRanges")
    params.delete("languages")
    params.delete("authors")
    params.delete("targets")
    params.delete("search")

    if (workloadFilters.onlyPassed) params.set("onlyPassed", "1")
    if (workloadFilters.devices.length) params.set("devices", workloadFilters.devices.join(","))
    if (Object.keys(workloadFilters.axisRanges).length) params.set("axisRanges", JSON.stringify(workloadFilters.axisRanges))
    if (solutionFilters.languages.length) params.set("languages", solutionFilters.languages.join(","))
    if (solutionFilters.authors.length) params.set("authors", solutionFilters.authors.join(","))
    if (solutionFilters.targets.length) params.set("targets", solutionFilters.targets.join(","))
    if (solutionFilters.search) params.set("search", solutionFilters.search)

    if (pinnedP != null) params.set("p", String(pinnedP.toFixed(2)))
    else params.delete("p")

    const selections = Array.from(visibleSolutions)
    if (selections.length) params.set("solutions", selections.join(","))
    else params.delete("solutions")

    if (focusedSolution) params.set("focus", focusedSolution)
    else params.delete("focus")

    const next = params.toString()
    if (next !== lastQueryRef.current) {
      lastQueryRef.current = next
      const newUrl = `${window.location.pathname}${next ? `?${next}` : ""}`
      window.history.replaceState(null, "", newUrl)
    }
  }, [workloadFilters, solutionFilters, pinnedP, visibleSolutions, focusedSolution])

  const chips = useMemo(() => {
    const entries: { label: string; onRemove?: () => void }[] = []
    if (workloadFilters.onlyPassed)
      entries.push({ label: "Only PASSED", onRemove: () => setWorkloadFilters((filters) => ({ ...filters, onlyPassed: false })) })
    for (const device of workloadFilters.devices)
      entries.push({ label: `Device:${device}`, onRemove: () => setWorkloadFilters((filters) => ({ ...filters, devices: filters.devices.filter((d) => d !== device) })) })
    for (const [axis, range] of Object.entries(workloadFilters.axisRanges))
      entries.push({
        label: `${axis}∈[${range.min},${range.max}]`,
        onRemove: () => setWorkloadFilters((filters) => {
          const nextRanges = { ...filters.axisRanges }
          delete nextRanges[axis]
          return { ...filters, axisRanges: nextRanges }
        }),
      })
    for (const language of solutionFilters.languages)
      entries.push({ label: `Lang:${language}`, onRemove: () => setSolutionFilters((filters) => ({ ...filters, languages: filters.languages.filter((value) => value !== language) })) })
    for (const author of solutionFilters.authors)
      entries.push({ label: `Author:${author}`, onRemove: () => setSolutionFilters((filters) => ({ ...filters, authors: filters.authors.filter((value) => value !== author) })) })
    for (const target of solutionFilters.targets)
      entries.push({ label: `Target:${target}`, onRemove: () => setSolutionFilters((filters) => ({ ...filters, targets: filters.targets.filter((value) => value !== target) })) })
    return entries
  }, [workloadFilters, solutionFilters])

  const toggleVisible = useCallback((name: string) => {
    setVisibleSolutions((previous) => {
      const updated = new Set(previous)
      if (updated.has(name)) {
        updated.delete(name)
        return updated
      }
      if (updated.size >= MAX_VISIBLE_SOLUTIONS) {
        toast({ title: "Too many lines", description: `Limit ${MAX_VISIBLE_SOLUTIONS} curves for clarity.`, variant: "destructive" })
        return previous
      }
      updated.add(name)
      return updated
    })
  }, [])

  const visibleCurves = useMemo(() => {
    const subset: Record<string, CurvesPayload["curves"][string]> = {}
    if (!curves) return subset
    for (const [name, points] of Object.entries(curves.curves)) {
      if (visibleSolutions.has(name)) subset[name] = points
    }
    return subset
  }, [curves, visibleSolutions])

  const counts = useMemo(
    () => ({ solutions: Object.keys(curves?.curves || {}).length, workloads: curves?.nWorkloads || 0 }),
    [curves],
  )

  const solutionsForList = useMemo(
    () =>
      Object.keys(curves?.curves || {})
        .map((name) => solutions.find((solution) => solution.name === name))
        .filter(Boolean) as Solution[],
    [curves, solutions],
  )

  const handleHoverP = useCallback((_value: number | null) => {
    // Placeholder to keep Win@p hover plumbing intact
  }, [])

  return (
    <section id="solutions-traces" className="space-y-6">
      <h2 className="text-2xl font-semibold mb-2">Solutions & Traces</h2>

      <SolutionsTracesToolbar
        onOpenWorkload={() => setWorkloadDrawerOpen(true)}
        onOpenSolution={() => setSolutionDrawerOpen(true)}
        chips={chips}
        counts={counts}
      />

      <div className="container py-6 space-y-6">
        <WinAtPCurves
          curves={curves?.curves || {}}
          visible={visibleSolutions}
          onHoverP={handleHoverP}
          onPinP={setPinnedP}
          pinnedP={pinnedP}
          setSortScores={setSortScores}
          headline={`n = ${counts.workloads} workloads. Baseline: per-group preferring FlashInfer else fastest.`}
          colorFor={colorFor}
        />

        <SolutionsList
          solutions={solutionsForList}
          visible={visibleSolutions}
          onToggle={toggleVisible}
          sortScores={sortScores}
          search={searchQuery}
          setSearch={setSearchQuery}
          onFocus={setFocusedSolution}
          correctness={curves?.correctness || {}}
          colorFor={colorFor}
          onSelectAll={() => {
            if (!curves) return
            const allNames = Object.keys(curves.curves)
            const allowed = allNames.slice(0, MAX_VISIBLE_SOLUTIONS)
            setVisibleSolutions(new Set(allowed))
            if (allNames.length > MAX_VISIBLE_SOLUTIONS)
              toast({ title: "Selection limited", description: `Selected first ${MAX_VISIBLE_SOLUTIONS} solutions.`, variant: "default" })
          }}
          onDeselectAll={() => setVisibleSolutions(new Set())}
          maxVisible={MAX_VISIBLE_SOLUTIONS}
          focused={focusedSolution}
        />

        <WorkloadsTable
          traces={traces}
          solutions={solutions}
          pinnedP={pinnedP}
          focusSolution={focusedSolution}
          axisKeyOrder={axisKeyOrder}
          workloadFilters={workloadFilters}
        />
      </div>

      <SideDrawer title="Workload Filters" open={isWorkloadDrawerOpen} onClose={() => setWorkloadDrawerOpen(false)} side="left">
        <div className="space-y-4">
          <div>
            <div className="font-medium mb-2">Correctness</div>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={workloadFilters.onlyPassed}
                onChange={(event) => setWorkloadFilters((filters) => ({ ...filters, onlyPassed: event.target.checked }))}
              />
              Only show PASSED workloads
            </label>
          </div>
          <div>
            <div className="font-medium mb-2">Devices</div>
            <div className="flex flex-wrap gap-2">
              {devices.map((device) => (
                <label key={device} className="inline-flex items-center gap-1 text-sm border rounded px-2 py-1">
                  <input
                    type="checkbox"
                    checked={workloadFilters.devices.includes(device)}
                    onChange={(event) =>
                      setWorkloadFilters((filters) => ({
                        ...filters,
                        devices: event.target.checked
                          ? [...filters.devices, device]
                          : filters.devices.filter((value) => value !== device),
                      }))
                    }
                  />
                  {device}
                </label>
              ))}
            </div>
          </div>
          <div>
            <div className="font-medium mb-2">Axis ranges</div>
            <div className="space-y-3">
              {axisKeyOrder.map((axis) => {
                const fullRange = axisRangesAll[axis]
                const currentRange = workloadFilters.axisRanges[axis] || fullRange
                return (
                  <div key={axis} className="flex items-center gap-2 text-sm">
                    <span className="w-28 font-mono">{axis}</span>
                    <input
                      type="number"
                      className="w-24 h-8 rounded border px-2"
                      value={currentRange.min}
                      min={fullRange.min}
                      max={fullRange.max}
                      onChange={(event) =>
                        setWorkloadFilters((filters) => ({
                          ...filters,
                          axisRanges: {
                            ...filters.axisRanges,
                            [axis]: { min: Number(event.target.value), max: currentRange.max },
                          },
                        }))
                      }
                    />
                    <span>to</span>
                    <input
                      type="number"
                      className="w-24 h-8 rounded border px-2"
                      value={currentRange.max}
                      min={fullRange.min}
                      max={fullRange.max}
                      onChange={(event) =>
                        setWorkloadFilters((filters) => ({
                          ...filters,
                          axisRanges: {
                            ...filters.axisRanges,
                            [axis]: { min: currentRange.min, max: Number(event.target.value) },
                          },
                        }))
                      }
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        setWorkloadFilters((filters) => {
                          const nextRanges = { ...filters.axisRanges }
                          delete nextRanges[axis]
                          return { ...filters, axisRanges: nextRanges }
                        })
                      }
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                )
              })}
            </div>
          </div>
          <div className="flex justify-end">
            <Button onClick={() => setWorkloadDrawerOpen(false)}>Apply</Button>
          </div>
        </div>
      </SideDrawer>

      <SideDrawer title="Solution Filters" open={isSolutionDrawerOpen} onClose={() => setSolutionDrawerOpen(false)} side="right">
        <div className="space-y-4">
          <div>
            <div className="font-medium mb-2">Language</div>
            <div className="flex flex-wrap gap-2">
              {Array.from(new Set(solutions.map((solution) => solution.spec.language)))
                .sort()
                .map((language) => (
                  <label key={language} className="inline-flex items-center gap-1 text-sm border rounded px-2 py-1">
                    <input
                      type="checkbox"
                      checked={solutionFilters.languages.includes(language)}
                      onChange={(event) =>
                        setSolutionFilters((filters) => ({
                          ...filters,
                          languages: event.target.checked
                            ? [...filters.languages, language]
                            : filters.languages.filter((value) => value !== language),
                        }))
                      }
                    />
                    {language}
                  </label>
                ))}
            </div>
          </div>
          <div>
            <div className="font-medium mb-2">Authors</div>
            <div className="flex flex-wrap gap-2">
              {Array.from(new Set(solutions.map((solution) => solution.author)))
                .sort()
                .map((author) => (
                  <label key={author} className="inline-flex items-center gap-1 text-sm border rounded px-2 py-1">
                    <input
                      type="checkbox"
                      checked={solutionFilters.authors.includes(author)}
                      onChange={(event) =>
                        setSolutionFilters((filters) => ({
                          ...filters,
                          authors: event.target.checked
                            ? [...filters.authors, author]
                            : filters.authors.filter((value) => value !== author),
                        }))
                      }
                    />
                    {author}
                  </label>
                ))}
            </div>
          </div>
          <div>
            <div className="font-medium mb-2">Targets</div>
            <div className="flex flex-wrap gap-2">
              {Array.from(new Set(solutions.flatMap((solution) => solution.spec.target_hardware)))
                .sort()
                .map((target) => (
                  <label key={target} className="inline-flex items-center gap-1 text-sm border rounded px-2 py-1">
                    <input
                      type="checkbox"
                      checked={solutionFilters.targets.includes(target)}
                      onChange={(event) =>
                        setSolutionFilters((filters) => ({
                          ...filters,
                          targets: event.target.checked
                            ? [...filters.targets, target]
                            : filters.targets.filter((value) => value !== target),
                        }))
                      }
                    />
                    {target}
                  </label>
                ))}
            </div>
          </div>
          <div>
            <div className="font-medium mb-2">Search</div>
            <input
              value={solutionFilters.search}
              onChange={(event) => setSolutionFilters((filters) => ({ ...filters, search: event.target.value }))}
              placeholder="Name or ID"
              className="w-full h-9 rounded border px-2"
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setSolutionFilters(initialSolutionFilters)}>
              Reset
            </Button>
            <Button onClick={() => setSolutionDrawerOpen(false)}>Apply</Button>
          </div>
        </div>
      </SideDrawer>
    </section>
  )
}
