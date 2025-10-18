import { Suspense } from "react"
import Link from "next/link"
import { Button } from "@flashinfer-bench/ui"
import { ArrowRight } from "lucide-react"
import { Card, CardContent, CardHeader } from "@flashinfer-bench/ui"
import { ModelCard } from "@/components/model-card"
import { LeaderboardSection } from "@/components/leaderboard-section"
import { getAllDefinitions, getAllModels, getSolutionsForDefinition, getTracesForDefinition } from "@/lib/data-loader"
import { computeFastAtPCurvesForAuthors, type BaselineConfig } from "@/lib/analytics"
import baselinesData from "@/data/baselines.json"
import { KernelsSection } from "./kernels"

export const dynamic = "force-static"
export const revalidate = false

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

      return {
        definition,
        solutions,
        traces,
        solutionCount: solutions.length,
        traceCount: traces.length,
        baseline,
      }
    })
  )

  const definitionsWithCounts = definitionEntries.map(({ definition, solutionCount, traceCount }) => ({
    ...definition,
    solutionCount,
    traceCount,
  }))

  const authorDatasets = definitionEntries
    .filter((entry) => entry.solutions.length > 0 && entry.traces.length > 0)
    .map((entry) => ({
      solutions: entry.solutions,
      traces: entry.traces,
      baseline: entry.baseline,
    }))

  const leaderboardData = computeFastAtPCurvesForAuthors({
    datasets: authorDatasets,
    sampleCount: 300,
  })

  return (
    <div className="flex flex-col">
      <LeaderboardSection
        data={leaderboardData}
        baselineLabel="Per-definition baselines"
      />

      {/* Models Section */}
      <section className="container space-y-6 py-8 md:py-12">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h2 className="text-3xl font-bold tracking-tight">Models</h2>
            <p className="text-muted-foreground">
              Explore model architectures and their kernel implementations
            </p>
          </div>
          <Button asChild variant="ghost">
            <Link href="/models">
              View all <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {models.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              href={`/models/${model.id}`}
            />
          ))}
          {models.length === 0 && (
            <>
              {[1, 2, 3].map(i => (
                <Card key={i} className="animate-pulse">
                  <CardHeader>
                    <div className="h-5 w-32 bg-muted rounded" />
                    <div className="h-4 w-48 bg-muted rounded mt-2" />
                  </CardHeader>
                  <CardContent>
                    <div className="h-4 w-24 bg-muted rounded" />
                  </CardContent>
                </Card>
              ))}
            </>
          )}
        </div>
      </section>

      {/* Kernels Section */}
      <Suspense fallback={<div className="container py-12 text-sm text-muted-foreground">Loading kernels…</div>}>
        <KernelsSection definitions={definitionsWithCounts} />
      </Suspense>
    </div>
  )
}
