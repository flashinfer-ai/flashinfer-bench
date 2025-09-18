"use client"

import { useCallback, useEffect, useRef } from "react"
import * as d3 from "d3"
import { Card, CardContent, CardHeader, CardTitle, Button } from "@flashinfer-bench/ui"
import { BarChart2, Pin as PinIcon, Undo2 } from "lucide-react"
import type { CurvePoint } from "@/lib/analytics"

export type WinAtPCurvesProps = {
  curves: Record<string, CurvePoint[]>
  visible: Set<string>
  onHoverP: (p: number | null) => void
  onPinP: (p: number | null) => void
  pinnedP: number | null
  setSortScores: (scores: Record<string, number>) => void
  headline?: string
  colorFor: (name: string) => string
}

export function WinAtPCurves({
  curves,
  visible,
  onHoverP,
  onPinP,
  pinnedP,
  setSortScores,
  headline,
  colorFor,
}: WinAtPCurvesProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const lastScoresRef = useRef<Record<string, number>>({})

  function shallowEqualScores(next: Record<string, number>, prev: Record<string, number>) {
    const nextKeys = Object.keys(next)
    const prevKeys = Object.keys(prev)
    if (nextKeys.length !== prevKeys.length) return false
    return nextKeys.every((key) => next[key] === prev[key])
  }

  const updateScores = useCallback((nextScores: Record<string, number>) => {
    if (!shallowEqualScores(nextScores, lastScoresRef.current)) {
      lastScoresRef.current = nextScores
      setSortScores(nextScores)
    }
  }, [setSortScores])

  useEffect(() => {
    const chartSize = { width: 1000, height: 360, marginLeft: 48, marginRight: 16, marginTop: 16, marginBottom: 36 }
    const xScale = d3.scaleLinear().domain([0, 1]).range([chartSize.marginLeft, chartSize.width - chartSize.marginRight])
    const yScale = d3.scaleLinear().domain([0, 100]).range([chartSize.height - chartSize.marginBottom, chartSize.marginTop])
    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()
    svg.attr("viewBox", `0 0 ${chartSize.width} ${chartSize.height}`)

    const xAxis = d3.axisBottom(xScale).ticks(6).tickFormat((d) => `${d}`)
    const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat((d) => `${d}%`)
    svg.append("g").attr("transform", `translate(0,${chartSize.height - chartSize.marginBottom})`).call(xAxis as any)
    svg.append("g").attr("transform", `translate(${chartSize.marginLeft},0)`).call(yAxis as any)

    const line = d3
      .line<CurvePoint>()
      .x((point) => xScale(point.p))
      .y((point) => yScale(point.percent))
      .curve(d3.curveStepAfter)

    const scoresAt = (pValue: number) => {
      const scores: Record<string, number> = {}
      for (const [name, points] of Object.entries(curves)) {
        if (!visible.has(name) || points.length === 0) continue
        const index = Math.round(pValue * (points.length - 1))
        scores[name] = points[index]?.percent ?? 0
      }
      return scores
    }

    for (const [name, points] of Object.entries(curves)) {
      if (!visible.has(name)) continue
      svg
        .append("path")
        .datum(points)
        .attr("fill", "none")
        .attr("stroke", colorFor(name))
        .attr("stroke-width", 1.8)
        .attr("opacity", 0.95)
        .attr("d", line as any)
        .append("title")
        .text(name)
    }

    const crosshair = svg.append("g")
    const verticalLine = crosshair
      .append("line")
      .attr("y1", chartSize.marginTop)
      .attr("y2", chartSize.height - chartSize.marginBottom)
      .attr("stroke", "#888")
      .attr("stroke-dasharray", "4,4")
      .style("display", "none")

    const overlay = svg
      .append("rect")
      .attr("x", chartSize.marginLeft)
      .attr("y", chartSize.marginTop)
      .attr("width", chartSize.width - chartSize.marginLeft - chartSize.marginRight)
      .attr("height", chartSize.height - chartSize.marginTop - chartSize.marginBottom)
      .attr("fill", "transparent")
      .style("cursor", pinnedP != null ? "default" : "crosshair")
      .on("mousemove", function (event) {
        const [mouseX] = d3.pointer(event as any)
        const pValue = Math.max(0, Math.min(1, xScale.invert(mouseX)))
        onHoverP(pValue)
        verticalLine.style("display", null).attr("x1", mouseX).attr("x2", mouseX)
        updateScores(scoresAt(pValue))
      })
      .on("mouseleave", function () {
        onHoverP(null)
        verticalLine.style("display", "none")
        updateScores(scoresAt(0.95))
      })
      .on("click", function (event) {
        const [mouseX] = d3.pointer(event as any)
        const pValue = Math.max(0, Math.min(1, xScale.invert(mouseX)))
        onPinP(pValue)
      })

    if (pinnedP != null) {
      const pinnedX = xScale(pinnedP)
      svg
        .append("line")
        .attr("x1", pinnedX)
        .attr("x2", pinnedX)
        .attr("y1", chartSize.marginTop)
        .attr("y2", chartSize.height - chartSize.marginBottom)
        .attr("stroke", "#0ea5e9")
        .attr("stroke-width", 2)
      updateScores(scoresAt(pinnedP))
    } else {
      updateScores(scoresAt(0.95))
    }

    return () => {
      overlay.on("mousemove", null).on("mouseleave", null).on("click", null)
      svg.selectAll("*").remove()
    }
  }, [curves, visible, pinnedP, colorFor, onHoverP, onPinP, updateScores])

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart2 className="h-4 w-4" />
            <CardTitle>Win@p</CardTitle>
            <span className="text-xs text-muted-foreground">Higher is better. Sampled p∈[0,1].</span>
          </div>
          {pinnedP != null && (
            <div className="flex items-center gap-2 text-sm">
              <PinIcon className="h-4 w-4 text-sky-500" />
              <span>p = {pinnedP.toFixed(2)}</span>
              <Button variant="ghost" size="sm" onClick={() => onPinP(null)}>
                <Undo2 className="h-4 w-4 mr-1" />Unpin
              </Button>
            </div>
          )}
        </div>
        {headline && <div className="text-xs text-muted-foreground">{headline}</div>}
      </CardHeader>
      <CardContent>
        <svg ref={svgRef} className="w-full h-auto" />
      </CardContent>
    </Card>
  )
}
