"use client"

import { Fragment, useEffect, useMemo, useRef, useState } from "react"
import * as d3 from "d3"
import { Card, CardContent, CardHeader, CardTitle, Button, HoverCard, HoverCardContent, HoverCardTrigger } from "@flashinfer-bench/ui"
import { Pin as PinIcon, Undo2, HelpCircle } from "lucide-react"
import { FastPLabel } from "@/components/fast-p-label"
import type { CurvePoint } from "@/lib/analytics"

const LEGEND_MAX_ITEMS = 10
const LEGEND_NAME_MAX_LENGTH = 14

export type ScoreboardEntry = {
  name: string
  percent: number
}

export type FastPCurvesProps = {
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

export function FastPCurves({
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
}: FastPCurvesProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const hintShownRef = useRef(false)
  const hideHintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [showPinHint, setShowPinHint] = useState(false)
  const [hoveredLegend, setHoveredLegend] = useState<string | null>(null)
  const legendItems = useMemo(() => {
    const items: Array<{ name: string; displayName: string; color: string }> = []
    for (const name of Array.from(visible)) {
      if (!curves[name]) continue
      const displayName =
        name.length > LEGEND_NAME_MAX_LENGTH ? `${name.slice(0, LEGEND_NAME_MAX_LENGTH)}...` : name
      items.push({ name, displayName, color: colorFor(name) })
      if (items.length >= LEGEND_MAX_ITEMS) break
    }
    return items
  }, [curves, visible, colorFor])

  const totalVisible = useMemo(
    () => Array.from(visible).filter((name) => Boolean(curves[name])).length,
    [curves, visible]
  )

  const remainingLegendCount = Math.max(totalVisible - legendItems.length, 0)
  const legendContainerRef = useRef<HTMLDivElement | null>(null)

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
      const isHighlighted = !hoveredLegend || hoveredLegend === name
      const strokeWidth = isHighlighted ? 2.4 : 1.2
      const strokeOpacity = hoveredLegend ? (isHighlighted ? 1 : 0.25) : 0.95
      svg
        .append("path")
        .datum(points)
        .attr("fill", "none")
        .attr("stroke", colorFor(name))
        .attr("stroke-width", strokeWidth)
        .attr("opacity", strokeOpacity)
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
  }, [curves, visible, pinnedP, colorFor, onHoverP, onPinP, hoveredLegend])

  return (
    <Card>
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <CardTitle>
              <span className="inline-flex items-baseline gap-1">
                <FastPLabel className="font-semibold" />
              </span>
            </CardTitle>
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
                  <FastPLabel /> measures the portion of workloads this solution is faster than p × baseline performance.
                </p>
                <a
                  href="/docs/fast_p"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs font-medium text-primary hover:underline"
                >
                  Read the full docs →
                </a>
              </HoverCardContent>
            </HoverCard>
          </div>
          <div
            className="flex flex-1 justify-center"
            ref={legendContainerRef}
            onMouseLeave={() => setHoveredLegend(null)}
          >
            {legendItems.length > 0 && (
              <div className="flex flex-wrap items-center justify-center gap-y-2 text-xs">
                {legendItems.map((item, index) => {
                  const isHovered = hoveredLegend === item.name
                  return (
                    <Fragment key={item.name}>
                      <span
                        className={`inline-flex items-center rounded-full bg-muted px-2 py-1 font-medium transition-colors ${isHovered ? "text-primary" : "text-foreground"}`}
                        title={item.name}
                        onMouseEnter={() => setHoveredLegend(item.name)}
                        onFocus={() => setHoveredLegend(item.name)}
                        onBlur={() => {
                          requestAnimationFrame(() => {
                            if (!legendContainerRef.current) return
                            if (!legendContainerRef.current.contains(document.activeElement)) {
                              setHoveredLegend(null)
                            }
                          })
                        }}
                        tabIndex={0}
                        role="button"
                      >
                        <span
                          className="h-2.5 w-2.5 rounded-full mr-1"
                          style={{ backgroundColor: item.color }}
                          aria-hidden="true"
                        />
                        <span className="whitespace-nowrap">{item.displayName}</span>
                      </span>
                      {index < legendItems.length - 1 && (
                        <span
                          className="inline-block h-1 w-3 flex-shrink-0 pointer-events-none"
                          aria-hidden="true"
                        />
                      )}
                    </Fragment>
                  )
                })}
                {remainingLegendCount > 0 && (
                  <span className="text-muted-foreground">
                    +{remainingLegendCount} more
                  </span>
                )}
              </div>
            )}
          </div>
          {pinnedP != null && (
            <div className="ml-auto flex items-center gap-2 text-sm">
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
