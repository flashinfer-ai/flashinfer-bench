import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { ModelCard } from "@/components/model-card"
import { getAllDefinitions, getAllModels, getSolutionsForDefinition, getTracesForDefinition } from "@/lib/data-loader"
import { KernelsSection } from "./kernels-section"

export default async function HomePage() {
  const [allDefinitions, models] = await Promise.all([getAllDefinitions(), getAllModels()])

  // Load counts for each definition
  const definitionsWithCounts = await Promise.all(
    allDefinitions.map(async (def) => {
      const [solutions, traces] = await Promise.all([
        getSolutionsForDefinition(def.name),
        getTracesForDefinition(def.name)
      ])
      return {
        ...def,
        solutionCount: solutions.length,
        traceCount: traces.length
      }
    })
  )

  return (
    <div className="flex flex-col">
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
      <KernelsSection definitions={definitionsWithCounts} />
    </div>
  )
}
