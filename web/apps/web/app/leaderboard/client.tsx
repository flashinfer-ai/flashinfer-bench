"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import Link from "next/link"
import * as Dialog from "@radix-ui/react-dialog"
import { Button, Tabs, TabsContent, TabsList, TabsTrigger } from "@flashinfer-bench/ui"
import { ChevronDown, ChevronRight, Crown, Eye, EyeOff, X } from "lucide-react"
import { FastPCurves, type ScoreboardEntry } from "@/components/fast-p-chart"
import { FastPLabel } from "@/components/fast-p-label"
import type {
  AuthorCorrectnessResponse,
  AuthorCurvesResponse,
  CurvePoint,
  CorrectnessSummary,
  CoverageStats,
} from "@/lib/analytics"
import type { Definition } from "@/lib/schemas"
import { cn } from "@flashinfer-bench/utils"

type DefinitionAuthorDetail = {
  definition: Definition
  curves: Record<string, CurvePoint[]>
  comparisonCounts: Record<string, number>
  totalComparisons: number
  totalWorkloads: number
  coverage: Record<string, CoverageStats>
  solutionNamesByAuthor: Record<string, string[]>
}

type DefinitionDetail = {
  definition: Definition
  curve: CurvePoint[]
  comparisonCount: number
  totalWorkloads: number
  coverage?: CoverageStats
  solutionNames: string[]
}

const DEFAULT_PIN = 0.95
const DEFAULT_VISIBLE = 5
const LIST_MAX_HEIGHT = 288 // 18rem

function sampleCurve(points: CurvePoint[] | undefined, p: number): number {
  if (!points || points.length === 0) return 0
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
  return points[lo]?.percent ?? 0
}

function calculateAUC(points: CurvePoint[] | undefined): number {
  if (!points || points.length === 0) return 0
  let area = 0
  for (let i = 1; i < points.length; i++) {
    const p1 = points[i - 1]
    const p2 = points[i]
    const width = p2.p - p1.p
    const avgHeight = (p1.percent + p2.percent) / 2
    area += width * avgHeight
  }
  return area / 100 // Convert from percentage to decimal
}

function buildScoreboard(curves: Record<string, CurvePoint[]>, p: number, excludedAuthors: Set<string>): ScoreboardEntry[] {
  const entries: ScoreboardEntry[] = []
  for (const [name, points] of Object.entries(curves)) {
    if (excludedAuthors.has(name)) continue
    entries.push({ name, percent: sampleCurve(points, p) })
  }
  return entries.sort((a, b) => {
    if (b.percent !== a.percent) return b.percent - a.percent
    return a.name.localeCompare(b.name)
  })
}

type LeaderboardClientProps = {
  fast: AuthorCurvesResponse
  correctness: AuthorCorrectnessResponse
  excludedAuthors: string[]
  baselineLabel: string
  initialPinnedP?: number
  definitionAuthorDetails: DefinitionAuthorDetail[]
}

export function LeaderboardClient({
  fast,
  correctness,
  excludedAuthors,
  baselineLabel,
  initialPinnedP = DEFAULT_PIN,
  definitionAuthorDetails,
}: LeaderboardClientProps) {
  const [pinnedP, setPinnedP] = useState<number | null>(initialPinnedP)
  const [isListExpanded, setIsListExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState<"rankings" | "fast">("rankings")
  const [selectedAuthor, setSelectedAuthor] = useState<string | null>(null)
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)
  const [highlightedAuthor, setHighlightedAuthor] = useState<string | null>(null)
  const [rankingSortColumn, setRankingSortColumn] = useState<"author" | "speedup" | "resolved">("speedup")
  const [rankingSortOrder, setRankingSortOrder] = useState<"asc" | "desc">("desc")

  const excludedSet = useMemo(() => new Set(excludedAuthors), [excludedAuthors])

  const authorDefinitionMap = useMemo(() => {
    const map = new Map<string, DefinitionDetail[]>()
    for (const detail of definitionAuthorDetails) {
      const { definition, curves, comparisonCounts, totalWorkloads, coverage, solutionNamesByAuthor } = detail
      for (const [author, curve] of Object.entries(curves)) {
        if (excludedSet.has(author)) continue
        const entries = map.get(author) ?? []
        entries.push({
          definition,
          curve,
          comparisonCount: comparisonCounts[author] ?? 0,
          coverage: coverage[author],
          totalWorkloads,
          solutionNames: solutionNamesByAuthor[author] ?? [],
        })
        map.set(author, entries)
      }
    }
    return map
  }, [definitionAuthorDetails, excludedSet])

  const initialScoreboard = useMemo(
    () => buildScoreboard(fast.curves, initialPinnedP, excludedSet),
    [fast.curves, initialPinnedP, excludedSet]
  )

  const [visibleAuthors, setVisibleAuthors] = useState<Set<string>>(
    () => new Set(initialScoreboard.slice(0, DEFAULT_VISIBLE).map((entry) => entry.name))
  )

  useEffect(() => {
    setVisibleAuthors((prev) => {
      if (prev.size === 0) return prev
      const filtered = new Set(Array.from(prev).filter((name) => !excludedSet.has(name) && Boolean(fast.curves[name])))
      if (filtered.size === prev.size) return prev
      if (filtered.size === 0) {
        return new Set(initialScoreboard.slice(0, DEFAULT_VISIBLE).map((entry) => entry.name))
      }
      return filtered
    })
  }, [fast.curves, initialScoreboard, excludedSet])

  useEffect(() => {
    if (!highlightedAuthor) return
    setVisibleAuthors((prev) => {
      if (prev.has(highlightedAuthor)) return prev
      const next = new Set(prev)
      next.add(highlightedAuthor)
      return next
    })
  }, [highlightedAuthor])

  useEffect(() => {
    if (!highlightedAuthor) return
    if (!fast.curves[highlightedAuthor]) {
      setHighlightedAuthor(null)
    }
  }, [fast.curves, highlightedAuthor])

  const pinnedTarget = pinnedP ?? initialPinnedP
  const scoreboard = useMemo(
    () => buildScoreboard(fast.curves, pinnedTarget, excludedSet),
    [fast.curves, pinnedTarget, excludedSet]
  )
  const totalWorkloads = fast.totalWorkloads ?? 0
  const coverageMap = fast.coverage ?? {}

  const formatCoverage = useCallback((stats?: CoverageStats | null) => {
    if (!stats) return null
    const attempted = stats.attempted ?? 0
    const total = stats.total ?? 0
    const percentValue = total > 0 ? (attempted / total) * 100 : 0
    return {
      percentText: percentValue.toFixed(1),
    }
  }, [])

  const champion = scoreboard[0]
  const championCoverageInfo = formatCoverage(champion ? coverageMap[champion.name] : undefined)
  const championPercent = champion ? champion.percent.toFixed(1) : "0.0"

  const selectedAuthorDefinitions = useMemo(() => {
    if (!selectedAuthor) return []
    const entries = authorDefinitionMap.get(selectedAuthor) ?? []
    return entries
      .map((entry) => ({
        ...entry,
        winPercent: sampleCurve(entry.curve, pinnedTarget),
      }))
      .sort((a, b) => b.winPercent - a.winPercent)
  }, [selectedAuthor, authorDefinitionMap, pinnedTarget])

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

      // Calculate hash to get initial color preference
      let hash = 0
      for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0
      const preferredIndex = hash % palette.length

      // Find which colors are already in use
      const usedColors = new Set(colorMap.values())

      // Try preferred color first
      let color = palette[preferredIndex]
      if (!usedColors.has(color)) {
        colorMap.set(name, color)
        return color
      }

      // If preferred color is taken, find the next available color
      for (let i = 0; i < palette.length; i++) {
        const candidateIndex = (preferredIndex + i) % palette.length
        const candidateColor = palette[candidateIndex]
        if (!usedColors.has(candidateColor)) {
          colorMap.set(name, candidateColor)
          return candidateColor
        }
      }

      // Fallback: if all colors are used, cycle through palette
      const fallbackIndex = colorMap.size % palette.length
      const fallbackColor = palette[fallbackIndex]
      colorMap.set(name, fallbackColor)
      return fallbackColor
    },
    [colorMap, palette]
  )

  const handleHoverP = useCallback((value: number | null) => {
    void value
  }, [])

  const handlePinP = useCallback((value: number | null) => {
    setPinnedP(value)
  }, [])

  const toggleAuthor = useCallback((name: string) => {
    setVisibleAuthors((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }, [])

  const showAll = useCallback(() => {
    const authors = Object.keys(fast.curves).filter((name) => !excludedSet.has(name))
    setVisibleAuthors(new Set(authors))
  }, [fast.curves, excludedSet])

  const clearAll = useCallback(() => {
    setVisibleAuthors(new Set())
  }, [])

  const openAuthorDetail = useCallback((author: string) => {
    setSelectedAuthor(author)
    setIsDrawerOpen(true)
  }, [])

  const closeDrawer = useCallback(() => {
    setIsDrawerOpen(false)
    setSelectedAuthor(null)
  }, [])

  const inspectHighlightedAuthor = useCallback(() => {
    if (!highlightedAuthor) return
    setIsListExpanded(true)
    openAuthorDetail(highlightedAuthor)
  }, [highlightedAuthor, openAuthorDetail])

  const pinnedLabel = pinnedTarget.toFixed(2)

  const correctnessByAuthor = useMemo(() => {
    const map: Record<string, CorrectnessSummary> = {}
    for (const entry of correctness.stats) {
      map[entry.author] = {
        total: entry.total,
        passed: entry.passed,
        incorrect: entry.incorrect,
        runtime_error: entry.runtime_error,
        other: entry.other,
      }
    }
    return map
  }, [correctness.stats])

  const rankingsData = useMemo(() => {
    const allAuthors = new Set([
      ...Object.keys(fast.curves),
      ...correctness.stats.map(s => s.author)
    ])

    return Array.from(allAuthors)
      .filter(author => !excludedSet.has(author))
      .map(author => {
        const curve = fast.curves[author]
        const avgSpeedup = calculateAUC(curve)

        const correctnessEntry = correctness.stats.find(s => s.author === author)
        const passRate = correctnessEntry && correctnessEntry.total > 0
          ? correctnessEntry.passed / correctnessEntry.total
          : 0

        return {
          author,
          avgSpeedup,
          passRate,
          totalTests: correctnessEntry?.total ?? 0,
          passedTests: correctnessEntry?.passed ?? 0,
        }
      })
  }, [fast.curves, correctness.stats, excludedSet])

  const sortedRankings = useMemo(() => {
    const sorted = [...rankingsData]
    sorted.sort((a, b) => {
      let compareValue = 0
      switch (rankingSortColumn) {
        case "author":
          compareValue = a.author.localeCompare(b.author)
          break
        case "speedup":
          compareValue = a.avgSpeedup - b.avgSpeedup
          break
        case "resolved":
          compareValue = a.passRate - b.passRate
          break
      }
      return rankingSortOrder === "asc" ? compareValue : -compareValue
    })
    return sorted
  }, [rankingsData, rankingSortColumn, rankingSortOrder])

  const handleSortColumn = useCallback((column: "author" | "speedup" | "resolved") => {
    if (rankingSortColumn === column) {
      setRankingSortOrder(prev => prev === "asc" ? "desc" : "asc")
    } else {
      setRankingSortColumn(column)
      setRankingSortOrder("desc")
    }
  }, [rankingSortColumn])

  return (
    <section>
      <div className="container space-y-6 py-6 md:py-8">
        <div className="space-y-2">
          <h2 className="text-3xl font-semibold tracking-tight">Leaderboard</h2>
          <p className="text-muted-foreground">
            Examine overall author performance across every kernel definition and workload.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as "rankings" | "fast")}
          className="space-y-6"
        >
          <TabsList className="w-fit">
            <TabsTrigger value="rankings">Rankings</TabsTrigger>
            <TabsTrigger value="fast">
              <FastPLabel className="font-medium" />
            </TabsTrigger>
          </TabsList>

          <TabsContent value="rankings" className="space-y-6">
            <div className="rounded-lg border bg-card/50">
              {sortedRankings.length === 0 ? (
                <p className="p-4 text-sm text-muted-foreground">No ranking data available.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="border-b bg-muted/50">
                      <tr>
                        <th className="px-4 py-3 text-left">
                          <button
                            type="button"
                            onClick={() => handleSortColumn("author")}
                            className="flex items-center gap-2 text-sm font-semibold hover:text-primary"
                          >
                            Author
                            {rankingSortColumn === "author" && (
                              <span className="text-xs">{rankingSortOrder === "asc" ? "↑" : "↓"}</span>
                            )}
                          </button>
                        </th>
                        <th className="px-4 py-3 text-left">
                          <button
                            type="button"
                            onClick={() => handleSortColumn("speedup")}
                            className="flex items-center gap-2 text-sm font-semibold hover:text-primary"
                          >
                            Avg Speedup
                            {rankingSortColumn === "speedup" && (
                              <span className="text-xs">{rankingSortOrder === "asc" ? "↑" : "↓"}</span>
                            )}
                          </button>
                        </th>
                        <th className="px-4 py-3 text-left">
                          <button
                            type="button"
                            onClick={() => handleSortColumn("resolved")}
                            className="flex items-center gap-2 text-sm font-semibold hover:text-primary"
                          >
                            % Resolved
                            {rankingSortColumn === "resolved" && (
                              <span className="text-xs">{rankingSortOrder === "asc" ? "↑" : "↓"}</span>
                            )}
                          </button>
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {sortedRankings.map((entry, index) => (
                        <tr
                          key={entry.author}
                          className="hover:bg-muted/50 cursor-pointer transition-colors"
                          onClick={() => openAuthorDetail(entry.author)}
                        >
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-3">
                              <span className="text-xs font-semibold text-muted-foreground">
                                {index + 1}.
                              </span>
                              <span
                                className="inline-flex h-6 w-1.5 flex-shrink-0 rounded-full"
                                style={{ backgroundColor: colorFor(entry.author) }}
                                aria-hidden="true"
                              />
                              <span className="font-medium">{entry.author}</span>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span className="text-sm">{(entry.avgSpeedup * 100).toFixed(1)}%</span>
                          </td>
                          <td className="px-4 py-3">
                            <div className="space-y-1">
                              <span className="text-sm">
                                {(entry.passRate * 100).toFixed(1)}%
                              </span>
                              <span className="ml-2 text-xs text-muted-foreground">
                                ({entry.passedTests}/{entry.totalTests})
                              </span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="fast" className="space-y-6">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Click on the chart to pin a different p and see the ranking at that threshold.</span>
            </div>
            <FastPCurves
              curves={fast.curves}
              visible={visibleAuthors}
              onHoverP={handleHoverP}
              onPinP={handlePinP}
              pinnedP={pinnedP}
              baselineLabel={baselineLabel}
              comparisonCount={totalWorkloads}
              baselineAvailable={totalWorkloads > 0}
              colorFor={colorFor}
              scoreboard={scoreboard}
              countLabel="workloads"
              highlighted={highlightedAuthor}
              onHighlightChange={setHighlightedAuthor}
              highlightContext="drawer"
              onInspectHighlighted={inspectHighlightedAuthor}
              correctness={correctnessByAuthor}
              hideBaselineLabel
            />

            <div className="rounded-lg border bg-card/50">
              <button
                type="button"
                onClick={() => setIsListExpanded((prev) => !prev)}
                className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm font-medium"
              >
                <span>
                  Author ranking for <FastPLabel className="font-medium" value={pinnedTarget.toFixed(2)} />
                </span>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  {scoreboard.length > 0 ? (
                    <>
                      <div className="flex items-center gap-2">
                        <span className="flex items-center gap-1 font-medium text-foreground">
                          <Crown className="h-3.5 w-3.5 text-amber-500" />
                          {champion?.name}
                        </span>
                      </div>
                      <span className="flex items-center gap-1">{championPercent}% win</span>
                      {championCoverageInfo && (
                        <span className="text-muted-foreground">
                          · {championCoverageInfo.percentText}% evaluated
                        </span>
                      )}
                    </>
                  ) : (
                    <span>No authors available</span>
                  )}
                  <ChevronDown className={cn("h-4 w-4 transition-transform", isListExpanded ? "rotate-180" : undefined)} />
                </div>
              </button>

              {isListExpanded && (
                <div className="border-t px-4 pb-2">
                  <div className="flex flex-wrap items-center gap-2 py-2 text-xs text-muted-foreground">
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline" onClick={() => setPinnedP(1.0)}>
                        Set p=1.0
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => setPinnedP(0.7)}>
                        Set p=0.7
                      </Button>
                      <Button size="sm" variant="outline" onClick={showAll}>
                        Show all
                      </Button>
                      <Button size="sm" variant="ghost" onClick={clearAll}>
                        Clear
                      </Button>
                    </div>
                  </div>
                  <div
                    className="divide-y overflow-y-auto border rounded-md"
                    style={{ maxHeight: LIST_MAX_HEIGHT }}
                  >
                    {scoreboard.map((entry, index) => {
                      const isActive = visibleAuthors.has(entry.name)
                      const percent = entry.percent.toFixed(1)
                      const coverageInfo = formatCoverage(coverageMap[entry.name])
                      const isSelected = isDrawerOpen && selectedAuthor === entry.name
                      const isHighlighted = highlightedAuthor === entry.name
                      const baseTextClass =
                        isActive || isSelected || isHighlighted ? "text-primary" : "text-muted-foreground"
                      return (
                        <div
                          key={entry.name}
                          className={cn(
                            "group flex w-full cursor-pointer items-center gap-2 rounded-md border border-transparent px-3 py-2 text-sm transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                            isActive || isSelected ? "bg-primary/5 text-primary" : "hover:bg-muted/60",
                            isHighlighted ? "border-primary/60" : ""
                          )}
                          role="button"
                          tabIndex={0}
                          onClick={() => openAuthorDetail(entry.name)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault()
                              openAuthorDetail(entry.name)
                            }
                          }}
                        >
                          <span
                            className="inline-flex h-8 w-1.5 flex-shrink-0 rounded-full"
                            style={{ backgroundColor: colorFor(entry.name) }}
                            aria-hidden="true"
                          />
                          <div className="flex flex-1 items-center justify-between gap-3">
                            <div className="flex items-center gap-3">
                              <span className="text-xs font-semibold text-muted-foreground">{index + 1}.</span>
                              <span className={cn("font-medium", baseTextClass)}>{entry.name}</span>
                            </div>
                            <div className="flex items-center gap-2 text-xs">
                              <span className={cn("flex items-center gap-1", baseTextClass)}>
                                {percent}% win
                              </span>
                              {coverageInfo && (
                                <span className="text-muted-foreground">
                                  · {coverageInfo.percentText}% evaluated
                                </span>
                              )}
                              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground group-hover:text-foreground" />
                            </div>
                          </div>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            className={cn(
                              "h-8 w-8 rounded-md",
                              isActive ? "text-primary hover:text-primary" : "text-muted-foreground hover:text-foreground"
                            )}
                            aria-pressed={isActive}
                            aria-label={isActive ? `Hide ${entry.name}` : `Show ${entry.name}`}
                            onClick={(event) => {
                              event.stopPropagation()
                              toggleAuthor(entry.name)
                            }}
                          >
                            {isActive ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                          </Button>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>

      <AuthorDetailDrawer
        open={isDrawerOpen}
        author={selectedAuthor}
        definitions={selectedAuthorDefinitions}
        pinnedTarget={pinnedTarget}
        onOpenChange={(open) => {
          if (!open) {
            closeDrawer()
          } else {
            setIsDrawerOpen(true)
          }
        }}
      />
    </section>
  )
}

type AuthorDetailDrawerProps = {
  open: boolean
  author: string | null
  definitions: Array<DefinitionDetail & { winPercent: number }>
  pinnedTarget: number
  onOpenChange: (open: boolean) => void
}

function AuthorDetailDrawer({ open, author, definitions, pinnedTarget, onOpenChange }: AuthorDetailDrawerProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-background/70 backdrop-blur-sm data-[state=open]:animate-overlay-in data-[state=closed]:animate-overlay-out" />
        <Dialog.Content className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md translate-x-full flex-col border-l bg-background shadow-xl focus:outline-none data-[state=open]:translate-x-0 data-[state=open]:animate-drawer-in data-[state=closed]:animate-drawer-out">
          <header className="flex items-center justify-between border-b px-6 py-4">
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">
                p = {pinnedTarget.toFixed(2)}
              </p>
              <Dialog.Title className="text-lg font-semibold">
                {author ?? "Author overview"}
              </Dialog.Title>
            </div>
            <Dialog.Close asChild>
              <button
                type="button"
                className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-transparent text-muted-foreground transition-colors hover:bg-muted/80 hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                aria-label="Close author details"
              >
                <X className="h-4 w-4" />
              </button>
            </Dialog.Close>
          </header>

          <div className="flex-1 overflow-y-auto px-6 py-4">
            {definitions.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No benchmarked definitions found for this author at the selected threshold.
              </p>
            ) : (
              <div className="space-y-3">
                {definitions.map(({ definition, curve, comparisonCount, totalWorkloads, solutionNames, coverage, winPercent }) => {
                  const solutionCount = solutionNames.length
                  const attempted = coverage?.attempted ?? comparisonCount
                  const totalSlots = coverage?.total ?? (solutionCount > 0 ? totalWorkloads * solutionCount : 0)
                  const percentValue = coverage?.percent ?? (totalSlots > 0 ? (attempted / totalSlots) * 100 : 0)
                  const coverageLabel = `${percentValue.toFixed(1)}% evaluated`
                  const metaPrimary = `${winPercent.toFixed(1)}% win · ${coverageLabel}`
                  const metaSecondary = solutionNames.length ? `${solutionNames.length} solutions` : null
                  return (
                    <Link
                      key={definition.name}
                      href={`/kernels/${definition.name}`}
                      className="flex items-center justify-between gap-3 rounded-lg border px-3 py-2 transition-colors hover:border-primary hover:bg-primary/5"
                    >
                      <div className="min-w-0 space-y-1">
                        <p className="truncate text-sm font-semibold">{definition.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {metaPrimary}
                          {metaSecondary && <br />}
                          {metaSecondary}
                        </p>
                      </div>
                      <Sparkline curve={curve} className="text-primary" />
                    </Link>
                  )
                })}
              </div>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

function Sparkline({ curve, className }: { curve: CurvePoint[]; className?: string }) {
  const width = 132
  const height = 40
  const padding = 6

  const path = useMemo(() => {
    if (!curve || curve.length === 0) return ""
    const usableWidth = width - padding * 2
    const usableHeight = height - padding * 2
    return curve
      .map((point, index) => {
        const x = padding + point.p * usableWidth
        const y = padding + (1 - Math.min(Math.max(point.percent, 0), 100) / 100) * usableHeight
        const command = index === 0 ? "M" : "L"
        return `${command}${x.toFixed(2)} ${y.toFixed(2)}`
      })
      .join(" ")
  }, [curve, height, padding, width])

  if (!path) {
    return (
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        aria-hidden="true"
        className={cn("text-muted-foreground", className)}
      >
        <line
          x1={padding}
          x2={width - padding}
          y1={height - padding}
          y2={height - padding}
          stroke="currentColor"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeDasharray="4 2"
        />
      </svg>
    )
  }

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden="true"
      className={cn("text-muted-foreground", className)}
    >
      <path d={path} fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" />
    </svg>
  )
}
