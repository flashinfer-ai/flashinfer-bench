"use client"

import { Fragment, useMemo, useState } from "react"
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Table, TableBody, TableCell, TableHead, TableHeader, TableRow, Tabs, TabsList, TabsTrigger } from "@flashinfer-bench/ui"
import { ListChecks } from "lucide-react"
import type { Solution, Trace } from "@/lib/schemas"
import { computeMeetsMisses, pickBaselineLatency, type WorkloadFilters } from "@/lib/analytics"

export type WorkloadsTableProps = {
  traces: Trace[]
  solutions: Solution[]
  pinnedP: number | null
  focusSolution: string | null
  axisKeyOrder: string[]
  workloadFilters: WorkloadFilters
}

export function WorkloadsTable({
  traces,
  solutions,
  pinnedP,
  focusSolution,
  axisKeyOrder,
  workloadFilters,
}: WorkloadsTableProps) {
  const [activeTab, setActiveTab] = useState<"meets" | "misses">("meets")
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})

  const meetsMisses = useMemo(() => {
    if (pinnedP == null || !focusSolution) return null
    return computeMeetsMisses({ traces, solutions, solutionName: focusSolution, p: pinnedP, workloadFilters })
  }, [traces, solutions, focusSolution, pinnedP, workloadFilters])

  const groups = meetsMisses ? (activeTab === "meets" ? meetsMisses.meets : meetsMisses.misses) : []
  const tracesByGroup: Record<string, Trace[]> = meetsMisses?.byGroup || {}

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ListChecks className="h-4 w-4" />
            <CardTitle>Traces</CardTitle>
          </div>
          {pinnedP != null && focusSolution && (
            <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as "meets" | "misses") }>
              <TabsList>
                <TabsTrigger value="meets">Meets</TabsTrigger>
                <TabsTrigger value="misses">Misses</TabsTrigger>
              </TabsList>
            </Tabs>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {pinnedP == null ? (
          <div className="text-sm text-muted-foreground">Pin a p on the Win@p chart.</div>
        ) : !focusSolution ? (
          <div className="text-sm text-muted-foreground">Choose a focus solution.</div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Workload (axes)</TableHead>
                <TableHead>Baseline</TableHead>
                <TableHead>Baseline Perf</TableHead>
                <TableHead>Trace</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {groups.map((groupId) => {
                const groupTraces = tracesByGroup[groupId] || []
                const representativeTrace = groupTraces[0]
                const axes = representativeTrace?.workload?.axes || {}
                const solutionMap = new Map(solutions.map((solution) => [solution.name, solution]))
                const baselineLatency = pickBaselineLatency(groupTraces, solutionMap) || undefined
                const baselineTrace = baselineLatency != null
                  ? groupTraces.find((trace) => trace.evaluation?.performance?.latency_ms === baselineLatency)
                  : undefined
                const focusedTrace = groupTraces.find((trace) => trace.solution === focusSolution)
                const status = focusedTrace?.evaluation?.status || "-"

                return (
                  <Fragment key={groupId}>
                    <TableRow
                      onClick={() => setExpanded((existing) => ({ ...existing, [groupId]: !existing[groupId] }))}
                      className="cursor-pointer"
                    >
                      <TableCell className="font-mono text-xs">
                        {axisKeyOrder.map((key) => (
                          <span key={key} className="mr-2">{key}={String((axes as Record<string, unknown>)[key])}</span>
                        ))}
                      </TableCell>
                      <TableCell className="font-mono text-xs">{baselineTrace?.solution || "-"}</TableCell>
                      <TableCell className="text-xs">{baselineLatency != null ? `${baselineLatency.toFixed(3)} ms` : "-"}</TableCell>
                      <TableCell className="text-xs">
                        <Button variant="outline" size="sm" onClick={(event) => { event.stopPropagation(); alert("Trace details preview coming soon.") }}>
                          Open trace
                        </Button>
                      </TableCell>
                      <TableCell>
                        <Badge variant={status === "PASSED" ? "secondary" : status.includes("ERROR") ? "destructive" : "outline"}>{status}</Badge>
                      </TableCell>
                    </TableRow>
                    {expanded[groupId] && (
                      <TableRow>
                        <TableCell colSpan={5}>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-2 bg-muted/40 rounded">
                            {groupTraces.map((trace, index) => {
                              const solution = solutions.find((candidate) => candidate.name === trace.solution)
                              return (
                                <div key={index} className="text-xs">
                                  <div className="flex items-center gap-2">
                                    <span className="font-mono">{trace.solution || "-"}</span>
                                    <span className="text-muted-foreground">{solution?.author || "-"}</span>
                                  </div>
                                  <div>Latency: {trace.evaluation?.performance?.latency_ms?.toFixed(3) ?? "-"} ms</div>
                                  <div>Speedup: {trace.evaluation?.performance?.speedup_factor?.toFixed(2) ?? "-"}x</div>
                                  <div>Max error: {trace.evaluation?.correctness?.max_relative_error?.toExponential(2) ?? "-"}</div>
                                  <div>Device: {trace.evaluation?.environment?.device ?? trace.evaluation?.environment?.hardware ?? "-"}</div>
                                </div>
                              )
                            })}
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </Fragment>
                )
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
