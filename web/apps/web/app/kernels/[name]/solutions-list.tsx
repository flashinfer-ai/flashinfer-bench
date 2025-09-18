"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, Button, DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@flashinfer-bench/ui"
import { Search, ChevronDown, Code2, Star } from "lucide-react"
import { cn } from "@flashinfer-bench/utils"
import type { Solution } from "@/lib/schemas"
import type { CorrectnessStats } from "./solutions-traces-types"

function middleTruncate(str: string, max = 24) {
  if (str.length <= max) return str
  const head = Math.ceil((max - 1) / 2)
  const tail = Math.floor((max - 1) / 2)
  return str.slice(0, head) + "…" + str.slice(-tail)
}

export type SolutionsListProps = {
  solutions: Solution[]
  visible: Set<string>
  onToggle: (name: string) => void
  sortScores: Record<string, number>
  search: string
  setSearch: (value: string) => void
  onFocus: (name: string) => void
  correctness: Record<string, CorrectnessStats>
  colorFor: (name: string) => string
  onSelectAll: () => void
  onDeselectAll: () => void
  maxVisible: number
  focused: string | null
}

export function SolutionsList({
  solutions,
  visible,
  onToggle,
  sortScores,
  search,
  setSearch,
  onFocus,
  correctness,
  colorFor,
  onSelectAll,
  onDeselectAll,
  maxVisible,
  focused,
}: SolutionsListProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const rowHeight = 72

  const filteredSolutions = useMemo(() => {
    const query = search.trim().toLowerCase()
    const matched = query
      ? solutions.filter((solution) => `${solution.name} ${solution.author}`.toLowerCase().includes(query))
      : solutions
    const hasScores = Object.keys(sortScores || {}).length > 0

    return [...matched].sort((a, b) => {
      if (hasScores) {
        const scoreA = sortScores[a.name] ?? -Infinity
        const scoreB = sortScores[b.name] ?? -Infinity
        if (scoreB !== scoreA) return scoreB - scoreA
      }

      const passedA = correctness[a.name]?.passed || 0
      const passedB = correctness[b.name]?.passed || 0
      if (passedB !== passedA) return passedB - passedA
      return a.name.localeCompare(b.name)
    })
  }, [solutions, search, correctness, sortScores])

  const handleScroll = useCallback(() => {
    const node = containerRef.current
    if (node) setScrollTop(node.scrollTop)
  }, [])

  const [viewportHeight, setViewportHeight] = useState(480)
  useEffect(() => {
    const updateViewport = () => {
      if (containerRef.current) {
        setViewportHeight(containerRef.current.clientHeight)
      }
    }
    updateViewport()
    window.addEventListener("resize", updateViewport)
    return () => window.removeEventListener("resize", updateViewport)
  }, [])

  const startIdx = Math.max(0, Math.floor(scrollTop / rowHeight) - 4)
  const endIdx = Math.min(filteredSolutions.length, Math.ceil((scrollTop + viewportHeight) / rowHeight) + 4)
  const visibleItems = filteredSolutions.slice(startIdx, endIdx)

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-2xl">Solutions</CardTitle>
          <div className="flex items-center gap-2">
            <div className="relative w-72">
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search solutions…"
                className="w-full rounded-md border bg-background pl-8 pr-2 h-9 text-sm"
              />
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  Bulk <ChevronDown className="h-4 w-4 ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem onClick={onSelectAll}>Select all (max {maxVisible})</DropdownMenuItem>
                <DropdownMenuItem onClick={onDeselectAll}>Deselect all</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div ref={containerRef} onScroll={handleScroll} className="relative h-[480px] overflow-auto border rounded-md">
          <div style={{ height: filteredSolutions.length * rowHeight }} />
          <div className="absolute inset-x-0" style={{ top: startIdx * rowHeight }}>
            {visibleItems.map((solution) => {
              const correctnessStats = correctness[solution.name]
              const total = correctnessStats?.total || 0
              const passedPct = total ? (correctnessStats.passed / total) * 100 : 0
              const incorrectPct = total ? (correctnessStats.incorrect / total) * 100 : 0
              const runtimeErrorPct = total ? (correctnessStats.runtime_error / total) * 100 : 0

              return (
                <div key={solution.name} className={cn("px-3 py-3 border-b", focused === solution.name && "bg-muted")}>
                  <div className="flex items-center gap-3 min-w-0">
                    <input
                      type="checkbox"
                      className="h-4 w-4"
                      checked={visible.has(solution.name)}
                      onChange={() => onToggle(solution.name)}
                      aria-label={`toggle ${solution.name}`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <button
                          onClick={() => onFocus(solution.name)}
                          className="text-left font-mono text-sm truncate"
                          title={solution.name}
                          aria-label={`focus ${solution.name}`}
                        >
                          {middleTruncate(solution.name, 40)}
                        </button>
                        <div className="flex items-center gap-2">
                          <button onClick={() => onFocus(solution.name)} aria-label={`focus ${solution.name}`}>
                            <Star className={cn("h-4 w-4", focused === solution.name && "text-yellow-500")} />
                          </button>
                          {visible.has(solution.name) && (
                            <button onClick={() => onFocus(solution.name)} aria-label={`focus ${solution.name}`}>
                              <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: colorFor(solution.name) }} />
                            </button>
                          )}
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => {
                              const solutionId = `${solution.definition}-${solution.name}`.replace(/[^a-zA-Z0-9-_]/g, "_")
                              sessionStorage.setItem(`solution-${solutionId}`, JSON.stringify(solution))
                              window.open(`/editor?solution=${solutionId}`, "_blank")
                            }}
                          >
                            <Code2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      <div className="text-[12px] text-muted-foreground truncate">
                        {solution.author}
                        <span className="mx-1">·</span>
                        {solution.spec.language}
                        <span className="mx-1">·</span>
                        {solution.spec.target_hardware.join(", ")}
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 h-2 w-full bg-muted rounded flex overflow-hidden relative">
                    <div className="bg-emerald-500" style={{ width: `${passedPct}%` }} />
                    <div className="bg-amber-500" style={{ width: `${incorrectPct}%` }} />
                    <div className="bg-rose-500" style={{ width: `${runtimeErrorPct}%` }} />
                    <div className="absolute -top-4 right-0 text-[11px] text-muted-foreground">{correctnessStats?.passed ?? 0}</div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
