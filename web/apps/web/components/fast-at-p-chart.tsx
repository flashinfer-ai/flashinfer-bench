"use client"

import { useEffect, useRef, useState } from "react"
import * as d3 from "d3"
import { Card, CardContent, CardHeader, CardTitle, Button, HoverCard, HoverCardContent, HoverCardTrigger } from "@flashinfer-bench/ui"
import { Pin as PinIcon, Undo2, HelpCircle } from "lucide-react"
import type { CurvePoint } from "@/lib/analytics"

export type ScoreboardEntry = {
  name: string
  percent: number
}

export type FastAtPCurvesProps = {
  curves: Record<string, CurvePoint[]>
  visible: Set<string>
  onHoverP: (p: number | null) => void
  onPinP: (p: number | null) => void
  pinnedP: number | null
  baselineLabel: string
  comparisonCount: number
  baselineAvailable: boolean
  colorFor: (name: string) => string
  scoreboard: ScoreboardEntry[]
  countLabel?: string
}

export function FastAtPCurves({
  curves,
  visible,
  onHoverP,
  onPinP,
  pinnedP,
  baselineLabel,
  comparisonCount,
  baselineAvailable,
  colorFor,
  scoreboard: _scoreboard,
  countLabel = "workloads",
}: FastAtPCurvesProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const hintShownRef = useRef(false)
  const hideHintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [showPinHint, setShowPinHint] = useState(false)

  useEffect(() => {
    if (pinnedP != null) {
      setShowPinHint(false)
      hintShownRef.current = true
      if (hideHintTimerRef.current) {
        clearTimeout(hideHintTimerRef.current)
        hideHintTimerRef.current = null
      }
    }
  }, [pinnedP])

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

        if (pinnedP == null && !hintShownRef.current) {
          hintShownRef.current = true
          setShowPinHint(true)
          if (hideHintTimerRef.current) clearTimeout(hideHintTimerRef.current)
          hideHintTimerRef.current = setTimeout(() => {
            setShowPinHint(false)
            hideHintTimerRef.current = null
          }, 2500)
        }
      })
      .on("mouseleave", function () {
        onHoverP(null)
        verticalLine.style("display", "none")
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
    }

    return () => {
      overlay.on("mousemove", null).on("mouseleave", null).on("click", null)
      svg.selectAll("*").remove()
    }
  }, [curves, visible, pinnedP, colorFor, onHoverP, onPinP])

  return (
    <Card>
      <CardHeader className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle>Fast@p Plot</CardTitle>
            <HoverCard>
              <HoverCardTrigger asChild>
                <button type="button" className="text-muted-foreground hover:text-foreground">
                  <HelpCircle className="h-4 w-4" />
                </button>
              </HoverCardTrigger>
              <HoverCardContent className="w-72 text-sm">
                <p className="text-xs font-medium text-primary">
                  What&apos;s this?
                </p>
                <p className="mb-2 text-sm text-muted-foreground">
                  Measures the portion of workloads this solution is faster than p × baseline performance.
                </p>
                <a
                  href="/docs/fast-at-p"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs font-medium text-primary hover:underline"
                >
                  Read the full docs →
                </a>
              </HoverCardContent>
            </HoverCard>
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
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>n = {comparisonCount} {countLabel}</span>
          <span>Baseline: {baselineLabel}</span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative">
          {showPinHint && pinnedP == null && baselineAvailable && (
            <div className="absolute right-4 top-4 z-10 rounded-md bg-background/95 px-3 py-2 text-xs shadow">
              Click to pin p
            </div>
          )}
          <div className={baselineAvailable ? undefined : "pointer-events-none opacity-40"}>
            <svg ref={svgRef} className="w-full h-auto" />
          </div>
          {!baselineAvailable && (
            <div className="absolute inset-0 flex items-center justify-center rounded-md bg-background/70 backdrop-blur-sm">
              <span className="text-sm text-muted-foreground">Baseline not available</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
