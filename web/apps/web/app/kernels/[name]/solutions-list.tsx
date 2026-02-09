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
import { ChevronDown, ChevronUp, Code2, Eye, EyeOff, Search, RotateCcw } from "lucide-react"
import type { Solution, Trace } from "@/lib/schemas"
import type { CorrectnessStats, SolutionFiltersState } from "./solutions-types"
import type { SolutionTraceBuckets, SolutionTraceComparison } from "@/lib/analytics"
import { FastPLabel } from "@/components/fast-p-label"
import { cn } from "@flashinfer-bench/utils"

const correctnessFallback: CorrectnessStats = {
  total: 0,
  passed: 0,
  incorrect: 0,
  runtime_error: 0,
  other: 0,
}

type DropdownCheckedState = boolean | "indeterminate"

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
              onCheckedChange={(checked: DropdownCheckedState) =>
                onToggle(option, checked === true)
              }
            >
              {option}
            </DropdownMenuCheckboxItem>
          ))
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function getSolutionElementId(name: string): string {
  return `solution-${name.replace(/[^a-zA-Z0-9_-]/g, "_")}`
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
  baselineSolutionName: string | null
  baselineComparisons: SolutionTraceComparison[] | null
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
  baselineSolutionName,
  baselineComparisons,
}: SolutionsListProps) {
  const baselineSolutions = baselineSolutionName
    ? solutions.filter((solution) => solution.name === baselineSolutionName)
    : []
  const otherSolutions = baselineSolutionName
    ? solutions.filter((solution) => solution.name !== baselineSolutionName)
    : solutions

  const renderSolutionCard = (solution: Solution, isBaseline: boolean) => {
    const stats = correctness[solution.name] ?? correctnessFallback
    const total = stats.total || 0
    const passed = stats.passed || 0
    const passPercent = total ? (passed / total) * 100 : 0
    const isVisible = visibleSolutions.has(solution.name)
    const isExpanded = expandedSolution === solution.name
    const color = !isBaseline && isVisible ? colorFor(solution.name) : "#d4d4d8"
    const bucketsForSolution = !isBaseline && isExpanded ? traceBuckets : null
    const baselineComparison = isBaseline && isExpanded ? baselineComparisons : null

    const handleOpenViewer = (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation()
      if (typeof window === "undefined") return
      const solutionId = `${solution.definition}-${solution.name}`.replace(/[^a-zA-Z0-9-_]/g, "_")
      window.sessionStorage.setItem(`solution-${solutionId}`, JSON.stringify(solution))
      window.open(`/viewer?solution=${encodeURIComponent(solutionId)}`, "_blank")
    }

    const elementId = getSolutionElementId(solution.name)

    return (
      <div key={solution.name} id={elementId} data-solution-name={solution.name} className="rounded-lg border">
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
            "flex w-full items-stretch gap-3 rounded-lg text-left transition-colors cursor-pointer",
            isExpanded ? "bg-muted/40" : "hover:bg-muted/20"
          )}
        >
          {!isBaseline && (
            <span
              className="w-1.5 rounded-l-lg"
              style={{ backgroundColor: color, opacity: isVisible ? 1 : 0.25 }}
            />
          )}
          <div className="flex flex-1 flex-col gap-4 px-4 py-3">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm">{solution.name}</span>
                  {isBaseline && (
                    <Badge variant="secondary" className="text-[10px] uppercase tracking-wide">
                      Baseline
                    </Badge>
                  )}
                </div>
                {!isBaseline ? (
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <Badge variant="outline" className="text-xs">
                      {solution.author}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {solution.spec.language}
                    </Badge>
                    {solution.spec.binding && (
                      <Badge variant="outline" className="text-xs">
                        C++ binding: {solution.spec.binding}
                      </Badge>
                    )}
                    {!solution.spec.destination_passing_style && (
                      <Badge variant="outline" className="text-xs">
                        DPS: off
                      </Badge>
                    )}
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
                ) : (
                  <p className="text-xs text-muted-foreground">{solution.description}</p>
                )}
              </div>
              <div className="flex items-center gap-2">
                {!isBaseline && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className={cn(
                      isVisible ? "text-primary hover:text-primary" : "text-muted-foreground hover:text-foreground"
                    )}
                    aria-pressed={isVisible}
                    aria-label={isVisible ? `Hide ${solution.name}` : `Show ${solution.name}`}
                    title={isVisible ? "Hide from chart" : "Show on chart"}
                    onClick={(event: MouseEvent<HTMLButtonElement>) => {
                      event.stopPropagation()
                      onToggleSolution(solution.name)
                    }}
                  >
                    {isVisible ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleOpenViewer}
                  aria-label={`Open ${solution.name} in viewer`}
                  title="View source code"
                >
                  <Code2 className="h-4 w-4" />
                </Button>
                <div className="text-muted-foreground">
                  {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </div>
              </div>
            </div>

            {!isBaseline && (
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
            )}
          </div>
        </div>

        {isExpanded && (
          isBaseline ? (
            <BaselineTraceDetails comparisons={baselineComparison} axisKeyOrder={axisKeyOrder} onOpenTrace={onOpenTrace} />
          ) : (
            <SolutionTraceDetails
              traceBuckets={bucketsForSolution}
              pinnedP={pinnedP}
              onPinDefault={onPinDefault}
              axisKeyOrder={axisKeyOrder}
              onOpenTrace={onOpenTrace}
            />
          )
        )}
      </div>
    )
  }

  return (
    <Card className="relative">
      <CardHeader className="pr-6">
        <div className="flex flex-col gap-4">
          <div>
            <CardTitle className="text-2xl">Solutions</CardTitle>
            <div className="mt-1 text-sm text-muted-foreground">
              Solutions: {stats.solutions}
              · Workloads: {stats.workloads}
            </div>
            {filterChips.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {filterChips.map((chip, idx) => (
                  <Badge
                    key={`${chip.label}-${idx}`}
                    variant="secondary"
                    className="gap-1"
                    onClick={(event: MouseEvent<HTMLDivElement>) =>
                      event.stopPropagation()
                    }
                  >
                    {chip.label}
                    {chip.onRemove && (
                      <button
                        className="ml-1 text-xs text-muted-foreground hover:text-foreground"
                        onClick={(event: MouseEvent<HTMLButtonElement>) => {
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
      <CardContent className="space-y-6">
        {baselineSolutions.length > 0 && (
          <div className="space-y-3">
            <SectionDivider label="Baseline" />
            {baselineSolutions.map((solution) => renderSolutionCard(solution, true))}
          </div>
        )}

        {otherSolutions.length > 0 && (
          <div className="space-y-3">
            {baselineSolutions.length > 0 && <SectionDivider label="Solutions" />}
            {otherSolutions.map((solution) => renderSolutionCard(solution, false))}
          </div>
        )}

        {baselineSolutions.length === 0 && otherSolutions.length === 0 && (
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
          onClick={(event: MouseEvent<HTMLButtonElement>) => {
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
    const pLabel = pinnedP.toFixed(2)
    return (
      <div className="border-t bg-muted/10 px-6 py-4">
        <Tabs value="faster">
          <TabsList>
            <TabsTrigger value="faster" disabled>
              Faster@p={pLabel} (0)
            </TabsTrigger>
            <TabsTrigger value="slower" disabled>
              Slower@p={pLabel} (0)
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

  return (
    <div className="border-t bg-muted/10 px-6 py-4">
      <Tabs
        value={tab}
        onValueChange={(value) => {
          if (value === "faster" || value === "slower" || value === "incorrect") {
            setTab(value)
          }
        }}
      >
        <TabsList>
          <TabsTrigger value="faster">
            Faster@p={pinnedP.toFixed(2)} ({counts.faster})
          </TabsTrigger>
          <TabsTrigger value="slower">Slower@p={pinnedP.toFixed(2)} ({counts.slower})</TabsTrigger>
          <TabsTrigger value="incorrect">Incorrect ({counts.incorrect})</TabsTrigger>
        </TabsList>
        <div className="mt-4">
          <TabsContent value="faster">
            <TraceTable rows={buckets.faster} axisKeyOrder={axisKeyOrder} onOpenTrace={onOpenTrace} />
          </TabsContent>
          <TabsContent value="slower">
            <TraceTable rows={buckets.slower} axisKeyOrder={axisKeyOrder} onOpenTrace={onOpenTrace} />
          </TabsContent>
          <TabsContent value="incorrect">
            <TraceTable rows={buckets.incorrect} axisKeyOrder={axisKeyOrder} onOpenTrace={onOpenTrace} />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}

type BaselineTraceDetailsProps = {
  comparisons: SolutionTraceComparison[] | null
  axisKeyOrder: string[]
  onOpenTrace: (trace: Trace) => void
}

function BaselineTraceDetails({ comparisons, axisKeyOrder, onOpenTrace }: BaselineTraceDetailsProps) {
  if (!comparisons || comparisons.length === 0) {
    return (
      <div className="border-t bg-muted/10 px-6 py-4">
        <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
          No baseline traces available.
        </div>
      </div>
    )
  }

  return (
    <div className="border-t bg-muted/10 px-6 py-4">
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Workload</TableHead>
              <TableHead>Baseline Perf (ms)</TableHead>
              <TableHead className="text-right" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {comparisons.map((entry) => {
              const trace = entry.baseline ?? entry.candidate ?? null
              const workloadLabel = formatAxesSignature(trace ?? undefined, axisKeyOrder)
              const latency = entry.baselineLatency ?? entry.candidateLatency ?? null

              return (
                <TableRow key={entry.workloadId}>
                  <TableCell className="max-w-[220px] truncate font-mono text-xs" title={workloadLabel}>
                    {workloadLabel}
                  </TableCell>
                  <TableCell>{latency != null ? latency.toFixed(3) : "-"}</TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={(event: MouseEvent<HTMLButtonElement>) => {
                        event.stopPropagation()
                        if (trace) onOpenTrace(trace)
                      }}
                      disabled={!trace}
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
    </div>
  )
}

type TraceTableProps = {
  rows: SolutionTraceComparison[]
  axisKeyOrder: string[]
  onOpenTrace: (trace: Trace) => void
}

function TraceTable({ rows, axisKeyOrder, onOpenTrace }: TraceTableProps) {
  if (!rows.length) {
    return (
      <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
        No traces in this category.
      </div>
    )
  }

  const formatError = (value?: number | null) => {
    if (value == null) return "-"
    if (!Number.isFinite(value)) return String(value)
    const absValue = Math.abs(value)
    if (absValue >= 1 || absValue === 0) return value.toFixed(3)
    return value.toExponential(2)
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Workload</TableHead>
            <TableHead>Baseline Perf (ms)</TableHead>
            <TableHead>This Solution (ms)</TableHead>
            <TableHead>Max Abs Err</TableHead>
            <TableHead>Max Rel Err</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((entry) => {
            const workloadLabel = formatAxesSignature(entry.candidate ?? entry.baseline, axisKeyOrder)
            const baselineLatency = entry.baselineLatency ?? null
            const candidateLatency = entry.candidateLatency ?? null
            const status = entry.candidate?.evaluation?.status
            const correctness = entry.candidate?.evaluation?.correctness
            const maxAbsError = correctness?.max_absolute_error ?? null
            const maxRelError = correctness?.max_relative_error ?? null

            return (
              <TableRow key={entry.workloadId}>
                <TableCell className="max-w-[220px] truncate font-mono text-xs" title={workloadLabel}>
                  {workloadLabel}
                </TableCell>
                <TableCell>{baselineLatency != null ? baselineLatency.toFixed(3) : "-"}</TableCell>
                <TableCell>{candidateLatency != null ? candidateLatency.toFixed(3) : "-"}</TableCell>
                <TableCell>{formatError(maxAbsError)}</TableCell>
                <TableCell>{formatError(maxRelError)}</TableCell>
                <TableCell>
                  <Badge variant={statusVariant(status)}>{status || "-"}</Badge>
                </TableCell>
                <TableCell className="text-right">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(event: MouseEvent<HTMLButtonElement>) => {
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

function SectionDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
      <span className="flex-1 border-t border-border" />
      <span>{label}</span>
      <span className="flex-1 border-t border-border" />
    </div>
  )
}
