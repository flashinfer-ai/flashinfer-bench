"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import Link from "next/link"
import * as Dialog from "@radix-ui/react-dialog"
import { Button, Tabs, TabsContent, TabsList, TabsTrigger } from "@flashinfer-bench/ui"
import { ChevronDown, ChevronRight, Crown, Eye, EyeOff, X } from "lucide-react"
import { FastPCurves, type ScoreboardEntry } from "@/components/fast-p-chart"
import { FastPLabel } from "@/components/fast-p-label"
import type { AuthorCorrectnessResponse, AuthorCurvesResponse, CurvePoint } from "@/lib/analytics"
import type { Definition } from "@/lib/schemas"
import { cn } from "@flashinfer-bench/utils"

type DefinitionAuthorDetail = {
  definition: Definition
  curves: Record<string, CurvePoint[]>
  comparisonCounts: Record<string, number>
  totalComparisons: number
  solutionNamesByAuthor: Record<string, string[]>
}

type DefinitionDetail = {
  definition: Definition
  curve: CurvePoint[]
  comparisonCount: number
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
  const [activeTab, setActiveTab] = useState<"fast" | "correctness">("fast")
  const [selectedAuthor, setSelectedAuthor] = useState<string | null>(null)
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)

  const excludedSet = useMemo(() => new Set(excludedAuthors), [excludedAuthors])

  const authorDefinitionMap = useMemo(() => {
    const map = new Map<string, DefinitionDetail[]>()
    for (const detail of definitionAuthorDetails) {
      const { definition, curves, comparisonCounts, solutionNamesByAuthor } = detail
      for (const [author, curve] of Object.entries(curves)) {
        if (excludedSet.has(author)) continue
        const entries = map.get(author) ?? []
        entries.push({
          definition,
          curve,
          comparisonCount: comparisonCounts[author] ?? 0,
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

  const pinnedTarget = pinnedP ?? initialPinnedP
  const scoreboard = useMemo(
    () => buildScoreboard(fast.curves, pinnedTarget, excludedSet),
    [fast.curves, pinnedTarget, excludedSet]
  )

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
      let hash = 0
      for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0
      const color = palette[hash % palette.length]
      colorMap.set(name, color)
      return color
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

  const setTopN = useCallback(
    (n: number) => {
      setVisibleAuthors(new Set(scoreboard.slice(0, n).map((entry) => entry.name)))
    },
    [scoreboard]
  )

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

  const handleLegendClick = useCallback((author: string) => {
    setIsListExpanded(true)
    openAuthorDetail(author)
  }, [openAuthorDetail])

  const pinnedLabel = pinnedTarget.toFixed(2)

  const correctnessRanking = useMemo(() => {
    return correctness.stats
      .filter((entry) => !excludedSet.has(entry.author))
      .map((entry) => {
        const passRate = entry.total > 0 ? entry.passed / entry.total : 0
        return {
          ...entry,
          passRate,
        }
      })
      .sort((a, b) => {
        if (b.passRate !== a.passRate) return b.passRate - a.passRate
        if (b.total !== a.total) return b.total - a.total
        return a.author.localeCompare(b.author)
      })
  }, [correctness, excludedSet])

  const maxPassRate = correctnessRanking.length > 0 ? correctnessRanking[0].passRate : 0

  return (
    <section>
      <div className="container space-y-6 py-6 md:py-8">
        <div className="space-y-2">
          <h2 className="text-3xl font-semibold tracking-tight">Leaderboard</h2>
          <p className="text-sm text-muted-foreground">
            Examine overall author performance across every kernel definition and workload.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as "fast" | "correctness")}
          className="space-y-6"
        >
          <TabsList className="w-fit">
            <TabsTrigger value="fast">
              <FastPLabel className="font-medium" />
            </TabsTrigger>
            <TabsTrigger value="correctness">Correctness</TabsTrigger>
          </TabsList>

          <TabsContent value="fast" className="space-y-6">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Pinned p: {pinnedLabel}</span>
              <span>Pin a different p on the chart to see the ranking at that threshold.</span>
            </div>
            <FastPCurves
              curves={fast.curves}
              visible={visibleAuthors}
              onHoverP={handleHoverP}
              onPinP={handlePinP}
              pinnedP={pinnedP}
              baselineLabel={baselineLabel}
              comparisonCount={fast.totalComparisons}
              baselineAvailable={fast.totalComparisons > 0}
              colorFor={colorFor}
              scoreboard={scoreboard}
              countLabel="comparisons"
              onLegendClick={handleLegendClick}
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
                          {scoreboard[0].name}
                        </span>
                      </div>
                      <span className="flex items-center gap-1">
                        {scoreboard[0].percent.toFixed(1)}% win
                      </span>
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
                      <Button size="sm" variant="outline" onClick={() => setTopN(DEFAULT_VISIBLE)}>
                        Plot top {DEFAULT_VISIBLE}
                      </Button>
                      <Button size="sm" variant="outline" onClick={showAll}>
                        Plot all
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
                      const comparisons = fast.comparisonCounts[entry.name] ?? 0
                      const isSelected = isDrawerOpen && selectedAuthor === entry.name
                      const baseTextClass = isActive || isSelected ? "text-primary" : "text-muted-foreground"
                      return (
                        <div
                          key={entry.name}
                          className={cn(
                            "group flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-sm transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                            isActive || isSelected ? "bg-primary/5 text-primary" : "hover:bg-muted/60"
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
                              {comparisons > 0 && (
                                <span className="hidden md:inline text-muted-foreground">
                                  {comparisons} comps
                                </span>
                              )}
                              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground group-hover:text-foreground" />
                            </div>
                          </div>
                          <button
                            type="button"
                            aria-pressed={isActive}
                            aria-label={isActive ? `Hide ${entry.name}` : `Show ${entry.name}`}
                            onClick={(event) => {
                              event.stopPropagation()
                              toggleAuthor(entry.name)
                            }}
                            className={cn(
                              "inline-flex h-7 w-7 items-center justify-center rounded-md border transition-colors",
                              isActive
                                ? "border-primary/40 bg-primary/10 text-primary"
                                : "border-transparent text-muted-foreground hover:bg-muted/70"
                            )}
                          >
                            {isActive ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                          </button>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="correctness" className="space-y-6">
            <div className="rounded-lg border bg-card/50 p-4">
              {correctnessRanking.length === 0 ? (
                <p className="text-sm text-muted-foreground">No correctness data available.</p>
              ) : (
                <div className="space-y-4">
                  {correctnessRanking.map((entry, index) => {
                    const percent = (entry.passRate * 100).toFixed(1)
                    const width = maxPassRate > 0 ? `${(entry.passRate / maxPassRate) * 100}%` : "0%"
                    return (
                      <div key={entry.author} className="space-y-2">
                        <div className="flex items-center justify-between text-sm font-medium">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-semibold text-muted-foreground">{index + 1}.</span>
                            <span>{entry.author}</span>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {percent}% pass ({entry.passed}/{entry.total})
                          </div>
                        </div>
                        <div className="h-2 rounded bg-muted">
                          <div
                            className="h-full rounded bg-primary"
                            style={{ width }}
                          />
                        </div>
                      </div>
                    )
                  })}
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
        <Dialog.Overlay className="fixed inset-0 z-40 bg-background/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l bg-background shadow-xl focus:outline-none">
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
                {definitions.map(({ definition, curve, comparisonCount, solutionNames, winPercent }) => (
                  <Link
                    key={definition.name}
                    href={`/kernels/${definition.name}`}
                    className="flex items-center justify-between gap-3 rounded-lg border px-3 py-2 transition-colors hover:border-primary hover:bg-primary/5"
                  >
                    <div className="min-w-0 space-y-1">
                      <p className="truncate text-sm font-semibold">{definition.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {winPercent.toFixed(1)}% win
                        {comparisonCount ? ` · ${comparisonCount} comparisons` : ""}
                        {solutionNames.length ? ` · ${solutionNames.length} solutions` : ""}
                      </p>
                    </div>
                    <Sparkline curve={curve} className="text-primary" />
                  </Link>
                ))}
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
