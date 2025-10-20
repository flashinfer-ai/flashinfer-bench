import { Suspense } from "react"
import { LeaderboardSection } from "@/app/leaderboard/section"
import { ModelsSection } from "@/app/models"
import { getAllDefinitions, getAllModels, getSolutionsForDefinition, getTracesForDefinition } from "@/lib/data-loader"
import type { BaselineConfig } from "@/lib/analytics"
import baselinesData from "@/data/baselines.json"
import { KernelsSection } from "./kernels"

export default async function HomePage() {
  const [allDefinitions, models] = await Promise.all([getAllDefinitions(), getAllModels()])

  // Load counts for each definition
  const definitionEntries = await Promise.all(
    allDefinitions.map(async (definition) => {
      const [solutions, traces] = await Promise.all([
        getSolutionsForDefinition(definition.name),
        getTracesForDefinition(definition.name)
      ])

      const rawBaseline = (baselinesData as Record<string, Record<string, string> | undefined>)[definition.name]
      const baseline: BaselineConfig | undefined = rawBaseline
        ? {
            default: rawBaseline.default,
            devices: Object.fromEntries(
              Object.entries(rawBaseline).filter(([key]) => key !== "default")
            ),
          }
        : undefined

      const baselineNamesSet = new Set<string>()
      if (baseline?.default) baselineNamesSet.add(baseline.default)
      if (baseline?.devices) {
        for (const value of Object.values(baseline.devices)) baselineNamesSet.add(value)
      }

      return {
        definition,
        solutions,
        traces,
        solutionCount: solutions.length,
        traceCount: traces.length,
        baseline,
        baselineNames: Array.from(baselineNamesSet),
      }
    })
  )

  const definitionsWithCounts = definitionEntries.map(({ definition, solutionCount, traceCount }) => ({
    ...definition,
    solutionCount,
    traceCount,
  }))

  const leaderboardEntries = definitionEntries.map((entry) => ({
    definition: entry.definition,
    solutions: entry.solutions,
    traces: entry.traces,
    baseline: entry.baseline,
    baselineNames: entry.baselineNames,
  }))

  return (
    <div className="flex flex-col">
      {/* <LeaderboardSection
        entries={leaderboardEntries}
        baselineLabel="Per-definition baselines"
      /> */}

      <ModelsSection models={models} />

      <Suspense fallback={<div className="container py-12 text-sm text-muted-foreground">Loading kernels…</div>}>
        <KernelsSection definitions={definitionsWithCounts} />
      </Suspense>
    </div>
  )
}
