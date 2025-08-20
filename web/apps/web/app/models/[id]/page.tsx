import { notFound } from "next/navigation"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"
import { getModel, getAllModels } from "@/lib/data-loader"
import { ModelTabs } from "./model-tabs"

export async function generateStaticParams() {
  const models = await getAllModels()
  return models.map((model) => ({
    id: model.id,
  }))
}

export default async function ModelDetailPage({ 
  params,
  searchParams
}: { 
  params: Promise<{ id: string }>
  searchParams: Promise<{ from?: string }>
}) {
  const { id } = await params
  const { from } = await searchParams
  const model = await getModel(id)
  
  if (!model) {
    notFound()
  }
  
  return (
    <div className="container py-8">
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Link 
            href={from === 'models' ? '/models' : '/'} 
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4 inline mr-1" />
            Back to {from === 'models' ? 'models' : 'home'}
          </Link>
        </div>
        
        <div>
          <h1 className="text-3xl font-bold mb-2">{model.name}</h1>
          {model.description && (
            <p className="text-muted-foreground">{model.description}</p>
          )}
        </div>
        
        <ModelTabs model={model} />
      </div>
    </div>
  )
}