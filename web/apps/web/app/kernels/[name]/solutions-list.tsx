"use client"

import { useEffect, useState, type MouseEvent } from "react"
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@flashinfer-bench/ui"
import { ChevronDown, ChevronUp, Code2, Search, RotateCcw } from "lucide-react"
import type { Solution, Trace } from "@/lib/schemas"
import type { CorrectnessStats, SolutionFiltersState } from "./solutions-types"
import type { SolutionTraceBuckets, SolutionTraceComparison } from "@/lib/analytics"
import { cn } from "@flashinfer-bench/utils"

const correctnessFallback: CorrectnessStats = {
  total: 0,
  passed: 0,
  incorrect: 0,
  runtime_error: 0,
  other: 0,
}

function statusVariant(status?: string | null) {
  if (!status) return "outline" as const
  if (status === "PASSED") return "secondary" as const
  if (status.includes("ERROR")) return "destructive" as const
  if (status.startsWith("INCORRECT")) return "destructive" as const
  return "outline" as const
}

function formatAxesSignature(trace: Trace | undefined, axisKeyOrder: string[]) {
  if (!trace) return "-"
  const axes = trace.workload?.axes || {}
  const keys = axisKeyOrder.length ? axisKeyOrder : Object.keys(axes)
  if (keys.length === 0) return "-"
  return keys
    .filter((key) => axes[key] !== undefined)
    .map((key) => `${key}=${String((axes as Record<string, unknown>)[key])}`)
    .join(" ")
}

export type FilterChip = {
  label: string
  onRemove?: () => void
}

type FilterDropdownProps = {
  label: string
  selections: string[]
  options: string[]
  onToggle: (value: string, checked: boolean) => void
}

function FilterDropdown({ label, selections, options, onToggle }: FilterDropdownProps) {
  const count = selections.length
  const disabled = options.length === 0
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="min-w-[150px] justify-between gap-2"
          disabled={disabled}
        >
          <span>
            {label}
            {count ? ` (${count})` : ""}
          </span>
          <ChevronDown className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-56" align="start">
        <DropdownMenuLabel className="text-xs text-muted-foreground">{label}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {options.length === 0 ? (
          <DropdownMenuLabel className="text-xs text-muted-foreground">No options available</DropdownMenuLabel>
        ) : (
          options.map((option) => (
            <DropdownMenuCheckboxItem
              key={option}
              checked={selections.includes(option)}
              onCheckedChange={(checked) => onToggle(option, Boolean(checked))}
            >
              {option}
            </DropdownMenuCheckboxItem>
          ))
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export type SolutionsListProps = {
  solutions: Solution[]
  visibleSolutions: Set<string>
  onToggleSolution: (name: string) => void
  onExpandSolution: (name: string) => void
  expandedSolution: string | null
  correctness: Record<string, CorrectnessStats>
  colorFor: (name: string) => string
  pinnedP: number | null
  onPinDefault: () => void
  traceBuckets: SolutionTraceBuckets | null
  axisKeyOrder: string[]
  filterChips: FilterChip[]
  onOpenTrace: (trace: Trace) => void
  stats: { solutions: number; workloads: number }
  filters: SolutionFiltersState
  onSearchChange: (value: string) => void
  onToggleLanguage: (language: string, checked: boolean) => void
  onToggleAuthor: (author: string, checked: boolean) => void
  onToggleTarget: (target: string, checked: boolean) => void
  onResetFilters: () => void
  availableLanguages: string[]
  availableAuthors: string[]
  availableTargets: string[]
}

export function SolutionsList({
  solutions,
  visibleSolutions,
  onToggleSolution,
  onExpandSolution,
  expandedSolution,
  correctness,
  colorFor,
  pinnedP,
  onPinDefault,
  traceBuckets,
  axisKeyOrder,
  filterChips,
  onOpenTrace,
  stats,
  filters,
  onSearchChange,
  onToggleLanguage,
  onToggleAuthor,
  onToggleTarget,
  onResetFilters,
  availableLanguages,
  availableAuthors,
  availableTargets,
}: SolutionsListProps) {
  return (
    <Card className="relative">
      <CardHeader className="pr-6">
        <div className="flex flex-col gap-4">
          <div>
            <CardTitle className="text-2xl">Solutions</CardTitle>
            <div className="mt-1 text-sm text-muted-foreground">
              Solutions: {stats.solutions} · Workloads: {stats.workloads}
            </div>
            {filterChips.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {filterChips.map((chip, idx) => (
                  <Badge
                    key={`${chip.label}-${idx}`}
                    variant="secondary"
                    className="gap-1"
                    onClick={(event) => event.stopPropagation()}
                  >
                    {chip.label}
                    {chip.onRemove && (
                      <button
                        className="ml-1 text-xs text-muted-foreground hover:text-foreground"
                        onClick={(event) => {
                          event.stopPropagation()
                          chip.onRemove?.()
                        }}
                        aria-label={`remove ${chip.label}`}
                      >
                        ×
                      </button>
                    )}
                  </Badge>
                ))}
              </div>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                value={filters.search}
                onChange={(event) => onSearchChange(event.target.value)}
                placeholder="Search solutions"
                className="h-9 w-64 rounded-md border bg-background pl-8 pr-2 text-sm"
              />
            </div>
            <FilterDropdown
              label="Languages"
              selections={filters.languages}
              options={availableLanguages}
              onToggle={onToggleLanguage}
            />
            <FilterDropdown
              label="Authors"
              selections={filters.authors}
              options={availableAuthors}
              onToggle={onToggleAuthor}
            />
            <FilterDropdown
              label="Targets"
              selections={filters.targets}
              options={availableTargets}
              onToggle={onToggleTarget}
            />
            <Button
              variant="ghost"
              size="icon"
              onClick={onResetFilters}
              title="Reset filters"
              aria-label="Reset filters"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {solutions.map((solution) => {
          const stats = correctness[solution.name] ?? correctnessFallback
          const total = stats.total || 0
          const passed = stats.passed || 0
          const passPercent = total ? (passed / total) * 100 : 0
          const isVisible = visibleSolutions.has(solution.name)
          const isExpanded = expandedSolution === solution.name
          const color = isVisible ? colorFor(solution.name) : "#d4d4d8"
          const bucketsForSolution = isExpanded ? traceBuckets : null
          const handleOpenEditor = (event: MouseEvent<HTMLButtonElement>) => {
            event.stopPropagation()
            if (typeof window === "undefined") return
            const solutionId = `${solution.definition}-${solution.name}`.replace(/[^a-zA-Z0-9-_]/g, "_")
            window.sessionStorage.setItem(`solution-${solutionId}`, JSON.stringify(solution))
            window.open(`/editor?solution=${encodeURIComponent(solutionId)}`, "_blank")
          }

          return (
            <div key={solution.name} className="rounded-lg border">
              <div
                role="button"
                tabIndex={0}
                onClick={() => onExpandSolution(solution.name)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault()
                    onExpandSolution(solution.name)
                  }
                }}
                className={cn(
                  "flex w-full items-stretch gap-3 rounded-lg text-left transition-colors",
                  isExpanded ? "bg-muted/40" : "hover:bg-muted/20"
                )}
              >
                <span
                  className="w-1.5 rounded-l-lg"
                  style={{ backgroundColor: color, opacity: isVisible ? 1 : 0.25 }}
                />
                <div className="flex flex-1 flex-col gap-4 px-4 py-3">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="flex flex-col gap-2">
                      <span className="font-mono text-sm">{solution.name}</span>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        <Badge variant="outline" className="text-xs">
                          {solution.author}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {solution.spec.language}
                        </Badge>
                        {solution.spec.target_hardware.slice(0, 3).map((target) => (
                          <Badge key={target} variant="outline" className="text-xs">
                            {target}
                          </Badge>
                        ))}
                        {solution.spec.target_hardware.length > 3 && (
                          <Badge variant="outline" className="text-xs">
                            +{solution.spec.target_hardware.length - 3}
                          </Badge>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={isVisible}
                        onClick={(event) => event.stopPropagation()}
                        onChange={() => onToggleSolution(solution.name)}
                        className="h-4 w-4"
                        aria-label={`toggle ${solution.name}`}
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={handleOpenEditor}
                        aria-label={`Open ${solution.name} in editor`}
                        title="Open in editor"
                      >
                        <Code2 className="h-4 w-4" />
                      </Button>
                      <div className="text-muted-foreground">
                        {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <HoverCard>
                      <HoverCardTrigger asChild>
                        <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
                          <div
                            className="absolute inset-y-0 left-0 bg-emerald-500"
                            style={{ width: `${passPercent}%`, opacity: isVisible ? 1 : 0.6 }}
                          />
                        </div>
                      </HoverCardTrigger>
                      <HoverCardContent className="w-64 text-xs">
                        <div className="space-y-1">
                          <div>Passed: {stats.passed}</div>
                          <div>Incorrect: {stats.incorrect}</div>
                          <div>Runtime error: {stats.runtime_error}</div>
                          <div>Other: {stats.other}</div>
                          <div>Total: {stats.total}</div>
                        </div>
                      </HoverCardContent>
                    </HoverCard>
                    <div className="flex justify-end text-xs text-muted-foreground">
                      <span>Passed {passed}/{total || "-"}</span>
                    </div>
                  </div>
                </div>
              </div>

              {isExpanded && (
                <SolutionTraceDetails
                  traceBuckets={bucketsForSolution}
                  pinnedP={pinnedP}
                  onPinDefault={onPinDefault}
                  axisKeyOrder={axisKeyOrder}
                  onOpenTrace={onOpenTrace}
                />
              )}
            </div>
          )
        })}

        {solutions.length === 0 && (
          <div className="rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground">
            No solutions match the current filters.
          </div>
        )}
      </CardContent>
    </Card>
  )
}

type SolutionTraceDetailsProps = {
  traceBuckets: SolutionTraceBuckets | null
  pinnedP: number | null
  onPinDefault: () => void
  axisKeyOrder: string[]
  onOpenTrace: (trace: Trace) => void
}

function SolutionTraceDetails({
  traceBuckets,
  pinnedP,
  onPinDefault,
  axisKeyOrder,
  onOpenTrace,
}: SolutionTraceDetailsProps) {
  const buckets = traceBuckets ?? { faster: [], slower: [], incorrect: [] }
  const counts = {
    faster: buckets.faster.length,
    slower: buckets.slower.length,
    incorrect: buckets.incorrect.length,
  }
  const fasterCount = counts.faster
  const slowerCount = counts.slower
  const incorrectCount = counts.incorrect
  const totalCount = fasterCount + slowerCount + incorrectCount
  const initialTab = fasterCount > 0 ? "faster" : slowerCount > 0 ? "slower" : "incorrect"
  const [tab, setTab] = useState<"faster" | "slower" | "incorrect">(initialTab)

  useEffect(() => {
    setTab(initialTab)
  }, [initialTab])

  useEffect(() => {
    if (tab === "faster" && fasterCount === 0 && slowerCount > 0) {
      setTab("slower")
      return
    }
    if (tab === "slower" && slowerCount === 0 && fasterCount > 0) {
      setTab("faster")
      return
    }
    if (tab !== "incorrect") {
      const current = tab === "faster" ? fasterCount : slowerCount
      if (current === 0 && incorrectCount > 0) {
        setTab("incorrect")
      }
    } else if (incorrectCount === 0) {
      setTab(fasterCount > 0 ? "faster" : "slower")
    }
  }, [tab, fasterCount, slowerCount, incorrectCount])

  if (pinnedP == null) {
    return (
      <div className="border-t px-6 py-4 text-sm text-muted-foreground">
        Pin a p on the chart to see traces.
        <Button
          variant="outline"
          size="sm"
          className="ml-3"
          onClick={(event) => {
            event.stopPropagation()
            onPinDefault()
          }}
        >
          Pin 0.95
        </Button>
      </div>
    )
  }

  if (totalCount === 0) {
    return (
      <div className="border-t bg-muted/10 px-6 py-4">
        <Tabs value="faster">
          <TabsList>
            <TabsTrigger value="faster" disabled>
              Faster (0)
            </TabsTrigger>
            <TabsTrigger value="slower" disabled>
              Slower (0)
            </TabsTrigger>
            <TabsTrigger value="incorrect" disabled>
              Incorrect (0)
            </TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="mt-4 rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
          No traces for this solution.
        </div>
      </div>
    )
  }

  const ratioLabel = `r ≥ ${pinnedP.toFixed(2)}`

  return (
    <div className="border-t bg-muted/10 px-6 py-4">
      <Tabs value={tab} onValueChange={(value) => setTab(value as any)}>
        <TabsList>
          <TabsTrigger value="faster">Faster ({counts.faster})</TabsTrigger>
          <TabsTrigger value="slower">Slower ({counts.slower})</TabsTrigger>
          <TabsTrigger value="incorrect">Incorrect ({counts.incorrect})</TabsTrigger>
        </TabsList>
        <div className="mt-4">
          <TabsContent value="faster">
            <TraceTable rows={buckets.faster} axisKeyOrder={axisKeyOrder} ratioLabel={ratioLabel} onOpenTrace={onOpenTrace} />
          </TabsContent>
          <TabsContent value="slower">
            <TraceTable rows={buckets.slower} axisKeyOrder={axisKeyOrder} ratioLabel={ratioLabel} onOpenTrace={onOpenTrace} />
          </TabsContent>
          <TabsContent value="incorrect">
            <TraceTable rows={buckets.incorrect} axisKeyOrder={axisKeyOrder} ratioLabel={ratioLabel} onOpenTrace={onOpenTrace} />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}

type TraceTableProps = {
  rows: SolutionTraceComparison[]
  axisKeyOrder: string[]
  ratioLabel: string
  onOpenTrace: (trace: Trace) => void
}

function TraceTable({ rows, axisKeyOrder, ratioLabel, onOpenTrace }: TraceTableProps) {
  if (!rows.length) {
    return (
      <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
        No traces in this category.
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Workload</TableHead>
            <TableHead>Baseline</TableHead>
            <TableHead>Baseline Perf (ms)</TableHead>
            <TableHead>This Solution (ms)</TableHead>
            <TableHead>{ratioLabel}</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((entry) => {
            const workloadLabel = formatAxesSignature(entry.candidate ?? entry.baseline, axisKeyOrder)
            const baselineLatency = entry.baselineLatency ?? null
            const candidateLatency = entry.candidateLatency ?? null
            const ratio = entry.ratio ?? null
            const status = entry.candidate?.evaluation?.status

            return (
              <TableRow key={entry.workloadId}>
                <TableCell className="max-w-[220px] truncate font-mono text-xs" title={workloadLabel}>
                  {workloadLabel}
                </TableCell>
                <TableCell className="font-mono text-xs">{entry.baseline?.solution || "-"}</TableCell>
                <TableCell>{baselineLatency != null ? baselineLatency.toFixed(3) : "-"}</TableCell>
                <TableCell>{candidateLatency != null ? candidateLatency.toFixed(3) : "-"}</TableCell>
                <TableCell>{ratio != null ? ratio.toFixed(3) : "-"}</TableCell>
                <TableCell>
                  <Badge variant={statusVariant(status)}>{status || "-"}</Badge>
                </TableCell>
                <TableCell className="text-right">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(event) => {
                      event.stopPropagation()
                      if (entry.candidate) onOpenTrace(entry.candidate)
                    }}
                    disabled={!entry.candidate}
                  >
                    Open trace
                  </Button>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}
