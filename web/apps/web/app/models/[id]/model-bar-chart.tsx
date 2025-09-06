"use client"

import { useEffect, useRef } from "react"
import * as d3 from "d3"
import { Model } from "@/lib/schemas"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface ModuleData {
  name: string
  value: number
  hasKernel: boolean
  type: string
}

export function ModelBarChart({ model }: { model: Model }) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current) return

    // Prepare data - calculate hypothetical compute times
    const moduleData: ModuleData[] = Object.entries(model.modules).map(([name, module]) => {
      // Base value depends on module type and whether it has a kernel
      let baseValue = 1
      if (module.type === "layer") {
        baseValue = module.definition ? 5 : 2
      } else if (module.type === "block") {
        baseValue = 0.5 // Blocks are just containers
      }

      return {
        name,
        value: baseValue * (module.count || 1),
        hasKernel: !!module.definition,
        type: module.type
      }
    }).filter(d => d.value > 0).sort((a, b) => b.value - a.value).slice(0, 20)

    // Clear previous content
    d3.select(svgRef.current).selectAll("*").remove()

    // Set dimensions
    const margin = { top: 20, right: 30, bottom: 40, left: 200 }
    const width = 800 - margin.left - margin.right
    const height = Math.max(400, moduleData.length * 25) - margin.top - margin.bottom

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)

    // Create scales
    const x = d3.scaleLinear()
      .domain([0, d3.max(moduleData, d => d.value) || 1])
      .range([0, width])

    const y = d3.scaleBand()
      .domain(moduleData.map(d => d.name))
      .range([0, height])
      .padding(0.1)

    // Create bars
    const bars = g.selectAll(".bar")
      .data(moduleData)
      .enter().append("g")
      .attr("class", "bar")

    // Add rectangles
    bars.append("rect")
      .attr("x", 0)
      .attr("y", d => y(d.name)!)
      .attr("width", d => x(d.value))
      .attr("height", y.bandwidth())
      .attr("fill", d => d.hasKernel ? "#f59e0b" : "#94a3b8")
      .attr("opacity", 0.8)
      .on("mouseover", function() {
        d3.select(this).attr("opacity", 1)
      })
      .on("mouseout", function() {
        d3.select(this).attr("opacity", 0.8)
      })

    // Add value labels
    bars.append("text")
      .attr("x", d => x(d.value) + 5)
      .attr("y", d => y(d.name)! + y.bandwidth() / 2)
      .attr("dy", "0.35em")
      .text(d => d.value.toFixed(1))
      .attr("font-size", "12px")
      .attr("fill", "#666")

    // Add y axis
    g.append("g")
      .call(d3.axisLeft(y))
      .selectAll("text")
      .attr("font-size", "12px")

    // Add x axis
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(5))

    // Add x axis label
    g.append("text")
      .attr("transform", `translate(${width / 2},${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Relative Compute Units")
      .attr("font-size", "12px")

  }, [model])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Module Compute Contribution</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <svg ref={svgRef} className="w-full" />
          <div className="flex gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-amber-500 rounded" />
              <span>Has kernel implementation</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-gray-400 rounded" />
              <span>No kernel implementation</span>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            Note: Values are estimated based on module type and count. Connect to actual profiling data for accurate measurements.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
