"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { Button, Tabs, TabsContent, TabsList, TabsTrigger } from "@flashinfer-bench/ui"
import { ChevronDown, Crown, Eye, EyeOff } from "lucide-react"
import { FastPCurves, type ScoreboardEntry } from "@/components/fast-p-chart"
import { FastPLabel } from "@/components/fast-p-label"
import type { AuthorCorrectnessResponse, AuthorCurvesResponse, CurvePoint } from "@/lib/analytics"
import { cn } from "@flashinfer-bench/utils"

const DEFAULT_PIN = 0.95
const DEFAULT_VISIBLE = 5
const LIST_MAX_HEIGHT = 224 // 14rem

function sampleCurve(points: CurvePoint[] | undefined, p: number): number {
  if (!points || points.length === 0) return 0
  const clamped = Math.max(0, Math.min(1, p))
  const index = Math.round(clamped * (points.length - 1))
  return points[index]?.percent ?? 0
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

type LeaderboardSectionProps = {
  fast: AuthorCurvesResponse
  correctness: AuthorCorrectnessResponse
  excludedAuthors: string[]
  baselineLabel: string
  initialPinnedP?: number
}

export function LeaderboardSection({ fast, correctness, excludedAuthors, baselineLabel, initialPinnedP = DEFAULT_PIN }: LeaderboardSectionProps) {
  const [pinnedP, setPinnedP] = useState<number | null>(initialPinnedP)
  const [isListExpanded, setIsListExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState<"fast" | "correctness">("fast")

  const excludedSet = useMemo(() => new Set(excludedAuthors), [excludedAuthors])

  const initialScoreboard = useMemo(() => buildScoreboard(fast.curves, initialPinnedP, excludedSet), [fast.curves, initialPinnedP, excludedSet])

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
  const { filteredPassed, filteredTotal } = useMemo(() => {
    let passed = 0
    let total = 0
    for (const entry of correctnessRanking) {
      passed += entry.passed
      total += entry.total
    }
    return { filteredPassed: passed, filteredTotal: total }
  }, [correctnessRanking])

  const overallPassRate = filteredTotal > 0 ? filteredPassed / filteredTotal : 0

  return (
    <section>
      <div className="container space-y-6 py-6 md:py-8">
        <div className="space-y-2">
          <h2 className="text-3xl font-semibold tracking-tight">Leaderboard</h2>
          <p className="text-muted-foreground">
            Examine overall author performance across all kernels.
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
            />

            <div className="rounded-lg border bg-card/50">
              <button
                type="button"
                onClick={() => setIsListExpanded((prev) => !prev)}
                className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm font-medium"
              >
                <span>Author ranking for <FastPLabel className="font-medium" value={pinnedTarget.toFixed(2)} /></span>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  {scoreboard.length > 0 ? (
                    <>
                      <div className="flex items-center gap-2">
                        <span
                          className="inline-flex h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: colorFor(scoreboard[0].name) }}
                          aria-hidden="true"
                        />
                        <span className="flex items-center gap-1 font-medium text-foreground">
                          <Crown className="h-3.5 w-3.5 text-amber-500" />
                          {scoreboard[0].name}
                        </span>
                      </div>
                      <span className="flex items-center gap-1">
                        {scoreboard[0].percent.toFixed(1)}% faster
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
                        Show top {DEFAULT_VISIBLE}
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
                      const comparisons = fast.comparisonCounts[entry.name] ?? 0
                      return (
                        <button
                          key={entry.name}
                          type="button"
                          onClick={() => toggleAuthor(entry.name)}
                          className={cn(
                            "flex w-full items-center justify-between gap-3 px-3 py-2 text-sm transition-colors",
                            isActive ? "bg-primary/5 text-primary" : "hover:bg-muted/60"
                          )}
                        >
                          <div className="flex items-center gap-3">
                            <span className="text-xs font-semibold text-muted-foreground">{index + 1}.</span>
                            <span
                              className="inline-flex h-2.5 w-2.5 rounded-full"
                              style={{ backgroundColor: colorFor(entry.name) }}
                              aria-hidden="true"
                            />
                            <span className="font-medium">{entry.name}</span>
                          </div>
                          <div className="flex items-center gap-3 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              {percent}%
                              faster
                            </span>
                            <span>{comparisons} comps</span>
                            {isActive ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                          </div>
                        </button>
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
    </section>
  )
}
