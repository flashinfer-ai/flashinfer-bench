import { Suspense } from "react"
import { notFound } from "next/navigation"
import { getDefinition, getSolutionsForDefinition, getTracesForDefinition, getAllDefinitions } from "@/lib/data-loader"
import { computeCorrectnessSummaryForSolutions, computeWinAtPCurvesForSolutions, type BaselineConfig } from "@/lib/analytics"
import baselinesData from "@/data/baselines.json"
import { DefinitionHeader } from "./header"
import { AxesSignatureSection } from "./axes-sig"
import { ConstraintsSection } from "./constraints"
import { DefinitionReference } from "./reference"
import { SolutionsSection } from "./solutions"

export async function generateStaticParams() {
  const definitions = await getAllDefinitions()
  return definitions.map((definition) => ({
    name: definition.name,
  }))
}

export default async function TraceDetailPage({
  params
}: {
  params: Promise<{ name: string }>
}) {
  const { name } = await params
  const definition = await getDefinition(name)

  if (!definition) {
    notFound()
  }

  const [solutions, traces] = await Promise.all([
    getSolutionsForDefinition(definition.name),
    getTracesForDefinition(definition.name)
  ])

  const baselineConfig = (baselinesData as Record<string, Record<string, string> | undefined>)[definition.name]
  const baseline: BaselineConfig | undefined = baselineConfig
    ? {
        default: baselineConfig.default,
        devices: Object.fromEntries(Object.entries(baselineConfig).filter(([key]) => key !== "default")),
      }
    : undefined

  const correctness = computeCorrectnessSummaryForSolutions(traces, solutions)
  const { curves, nWorkloads } = computeWinAtPCurvesForSolutions({
    traces,
    solutions,
    baseline,
    sampleCount: 300,
  })

  const precomputed = {
    curves,
    correctness,
    nWorkloads,
  }

  return (
    <div className="relative">
      <DefinitionHeader
        definition={definition}
        solutionsCount={solutions.length}
      />

      <div className="container py-8">
        <div className="space-y-8">
          <p className="text-muted-foreground">{definition.description}</p>

          <AxesSignatureSection definition={definition} />

          <ConstraintsSection definition={definition} />

          <section id="reference">
            <h2 className="text-2xl font-semibold mb-4">Reference Implementation</h2>
            <DefinitionReference definition={definition} />
          </section>

          <Suspense fallback={<div className="py-8 text-sm text-muted-foreground">Loading solutions…</div>}>
            <SolutionsSection
              definition={definition}
              solutions={solutions}
              traces={traces}
              precomputed={precomputed}
            />
          </Suspense>
        </div>
      </div>
    </div>
  )
}
