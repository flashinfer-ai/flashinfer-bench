import { ModelCard } from "@/components/model-card"
import { Layers } from "lucide-react"
import { getAllModels } from "@/lib/data-loader"

export default async function ModelsPage() {
  const models = await getAllModels()
  
  return (
    <div className="container py-8">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Models</h1>
          <p className="text-muted-foreground">
            Explore model architectures and their kernel implementations
          </p>
        </div>
        
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {models.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              href={`/models/${model.id}?from=models`}
            />
          ))}
        </div>
        
        {models.length === 0 && (
          <div className="text-center py-12">
            <Layers className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No models found</p>
          </div>
        )}
      </div>
    </div>
  )
}