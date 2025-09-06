"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Trace, Solution, CanonicalWorkload } from "@/lib/schemas"

interface TracesTableProps {
  traces: Trace[]
  solutions: Solution[]
  canonicalWorkloads: CanonicalWorkload[]
}

export function TracesTable({ traces, solutions, canonicalWorkloads }: TracesTableProps) {
  const [selectedWorkload, setSelectedWorkload] = useState(canonicalWorkloads[0]?.name || "all")
  const [sortBy, setSortBy] = useState<"latency" | "speedup" | "name">("latency")

  // Get unique workload configurations
  const workloadConfigs = Array.from(new Set(
    traces.map(t => JSON.stringify(t.workload.axes))
  )).map(s => JSON.parse(s))

  // Filter traces based on selected workload
  const filteredTraces = selectedWorkload === "all"
    ? traces
    : traces.filter(t => {
        const workload = canonicalWorkloads.find(w => w.name === selectedWorkload)
        if (!workload) return false
        return Object.entries(workload.axes).every(
          ([k, v]) => t.workload.axes[k] === v
        )
      })

  // Sort traces
  const sortedTraces = [...filteredTraces].sort((a, b) => {
    if (sortBy === "latency") {
      return (a.evaluation?.performance?.latency_ms || 0) - (b.evaluation?.performance?.latency_ms || 0)
    } else if (sortBy === "speedup") {
      return (b.evaluation?.performance?.speedup_factor || 0) - (a.evaluation?.performance?.speedup_factor || 0)
    } else {
      return (a.solution || "").localeCompare(b.solution || "")
    }
  })

  // Group by workload config if showing all
  const groupedTraces = selectedWorkload === "all"
    ? workloadConfigs.map(config => ({
        config,
        traces: sortedTraces.filter(t => JSON.stringify(t.workload.axes) === JSON.stringify(config))
      })).filter(g => g.traces.length > 0)
    : [{ config: null, traces: sortedTraces }]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Traces</CardTitle>
        <CardDescription>
          Performance measurements across different workloads and solutions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Workload:</span>
            <Select value={selectedWorkload} onValueChange={setSelectedWorkload}>
              <SelectTrigger className="w-[250px]">
                <SelectValue placeholder="Select workload" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Workloads</SelectItem>
                {canonicalWorkloads.map(w => (
                  <SelectItem key={w.name} value={w.name}>
                    {w.description}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Select value={sortBy} onValueChange={(v) => setSortBy(v as any)}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="latency">Latency (ascending)</SelectItem>
              <SelectItem value="speedup">Speedup (descending)</SelectItem>
              <SelectItem value="name">Solution name</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-4">
          {groupedTraces.map(({ config, traces }, idx) => (
            <div key={idx} className="space-y-2">
              {config && (
                <h4 className="font-medium text-sm">
                  {Object.entries(config).map(([k, v]) => `${k}=${v}`).join(", ")}
                </h4>
              )}
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Solution</TableHead>
                    <TableHead>Author</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Latency (ms)</TableHead>
                    <TableHead>Speedup</TableHead>
                    <TableHead>Max Error</TableHead>
                    <TableHead>Device</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {traces.map((trace, i) => {
                    const solution = solutions.find(s => s.name === trace.solution)
                    return (
                      <TableRow key={i}>
                        <TableCell className="font-mono">{trace.solution || "-"}</TableCell>
                        <TableCell>{solution?.author || "-"}</TableCell>
                        <TableCell>
                          <Badge variant={trace.evaluation?.status === "PASSED" ? "default" : "destructive"}>
                            {trace.evaluation?.status || "PENDING"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {trace.evaluation?.performance?.latency_ms?.toFixed(3) || "-"}
                        </TableCell>
                        <TableCell>
                          {trace.evaluation?.performance?.speedup_factor
                            ? `${trace.evaluation.performance.speedup_factor.toFixed(2)}x`
                            : "-"}
                        </TableCell>
                        <TableCell className="text-xs">
                          {trace.evaluation?.correctness?.max_relative_error?.toExponential(2) || "-"}
                        </TableCell>
                        <TableCell className="text-xs">
                          {trace.evaluation?.environment.device || "-"}
                        </TableCell>
                      </TableRow>
                    )
                  })}
                  {traces.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center text-muted-foreground">
                        No traces found for this workload
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
