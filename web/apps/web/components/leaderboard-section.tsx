"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { Button } from "@flashinfer-bench/ui"
import { ChevronDown, Crown, Eye, EyeOff } from "lucide-react"
import { FastAtPCurves, type ScoreboardEntry } from "@/components/fast-at-p-chart"
import type { AuthorCurvesResponse, CurvePoint } from "@/lib/analytics"
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

function buildScoreboard(curves: Record<string, CurvePoint[]>, p: number): ScoreboardEntry[] {
  const entries: ScoreboardEntry[] = []
  for (const [name, points] of Object.entries(curves)) {
    entries.push({ name, percent: sampleCurve(points, p) })
  }
  return entries.sort((a, b) => {
    if (b.percent !== a.percent) return b.percent - a.percent
    return a.name.localeCompare(b.name)
  })
}

type LeaderboardSectionProps = {
  data: AuthorCurvesResponse
  baselineLabel: string
  initialPinnedP?: number
}

export function LeaderboardSection({ data, baselineLabel, initialPinnedP = DEFAULT_PIN }: LeaderboardSectionProps) {
  const [pinnedP, setPinnedP] = useState<number | null>(initialPinnedP)
  const [isListExpanded, setIsListExpanded] = useState(false)

  const initialScoreboard = useMemo(() => buildScoreboard(data.curves, initialPinnedP), [data.curves, initialPinnedP])

  const [visibleAuthors, setVisibleAuthors] = useState<Set<string>>(
    () => new Set(initialScoreboard.slice(0, DEFAULT_VISIBLE).map((entry) => entry.name))
  )

  useEffect(() => {
    setVisibleAuthors((prev) => {
      if (prev.size === 0) return prev
      const filtered = new Set(Array.from(prev).filter((name) => Boolean(data.curves[name])))
      if (filtered.size === prev.size) return prev
      if (filtered.size === 0) {
        return new Set(initialScoreboard.slice(0, DEFAULT_VISIBLE).map((entry) => entry.name))
      }
      return filtered
    })
  }, [data.curves, initialScoreboard])

  const pinnedTarget = pinnedP ?? initialPinnedP
  const scoreboard = useMemo(
    () => buildScoreboard(data.curves, pinnedTarget),
    [data.curves, pinnedTarget]
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
    setVisibleAuthors(new Set(Object.keys(data.curves)))
  }, [data.curves])

  const clearAll = useCallback(() => {
    setVisibleAuthors(new Set())
  }, [])

  const pinnedLabel = pinnedTarget.toFixed(2)

  if (data.totalComparisons === 0) {
    return null
  }

  return (
    <section>
      <div className="container space-y-6 py-10 md:py-14">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <h2 className="text-3xl font-semibold tracking-tight">Leaderboard</h2>
          </div>
          <div className="space-y-1 text-sm text-muted-foreground text-right">
            <p>Pinned p: {pinnedLabel}</p>
            <p className="text-xs text-muted-foreground">Pin a different p on the chart to see the ranking at that threshold.</p>
          </div>
        </div>

        <FastAtPCurves
          curves={data.curves}
          visible={visibleAuthors}
          onHoverP={handleHoverP}
          onPinP={handlePinP}
          pinnedP={pinnedP}
          baselineLabel={baselineLabel}
          comparisonCount={data.totalComparisons}
          baselineAvailable={data.totalComparisons > 0}
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
            <span>Author Ranking</span>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              {scoreboard.length > 0 ? (
                <>
                  <div className="flex items-center gap-2">
                    <span className="flex items-center gap-1 font-medium text-foreground">
                      <Crown className="h-3.5 w-3.5 text-amber-500" />
                      {scoreboard[0].name}
                    </span>
                  </div>
                  <span>{scoreboard[0].percent.toFixed(1)}%</span>
                </>
              ) : (
                <span>No authors available</span>
              )}
              <ChevronDown className={cn("h-4 w-4 transition-transform", isListExpanded ? "rotate-180" : undefined)} />
            </div>
          </button>

          {isListExpanded && (
            <div className="border-t px-4 pb-4">
              <div className="flex flex-wrap items-center gap-2 py-3 text-xs text-muted-foreground">
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
                  const comparisons = data.comparisonCounts[entry.name] ?? 0
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
                        />
                        <span className="font-medium">{entry.name}</span>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-muted-foreground">
                        <span>{percent}% Faster</span>
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
      </div>
    </section>
  )
}
