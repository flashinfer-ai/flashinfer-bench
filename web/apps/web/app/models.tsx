import Link from "next/link"
import { Button, Card, CardContent, CardHeader } from "@flashinfer-bench/ui"
import { ArrowRight } from "lucide-react"
import { ModelCard } from "@/components/model-card"
import type { Model } from "@/lib/schemas"

type ModelsSectionProps = {
  models: Model[]
}

export function ModelsSection({ models }: ModelsSectionProps) {
  const showViewAll = models.length > 3

  return (
    <section className="container space-y-6 py-8 md:py-12">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-3xl font-bold tracking-tight">Models</h2>
          <p className="text-muted-foreground">
            Explore model architectures and their kernel implementations
          </p>
        </div>
        {showViewAll && (
          <Button asChild variant="ghost">
            <Link href="/models">
              View all <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        )}
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
            {[1, 2, 3].map((i) => (
              <Card key={i} className="animate-pulse">
                <CardHeader>
                  <div className="h-5 w-32 rounded bg-muted" />
                  <div className="mt-2 h-4 w-48 rounded bg-muted" />
                </CardHeader>
                <CardContent>
                  <div className="h-4 w-24 rounded bg-muted" />
                </CardContent>
              </Card>
            ))}
          </>
        )}
      </div>
    </section>
  )
}
