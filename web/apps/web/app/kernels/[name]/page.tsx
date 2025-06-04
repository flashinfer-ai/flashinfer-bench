import { notFound } from "next/navigation"
import { getDefinition, getSolutionsForDefinition, getTracesForDefinition, getCanonicalWorkloads, getAllDefinitions } from "@/lib/data-loader"
import { DefinitionPageContent } from "./definition-page-content"

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
  
  const [solutions, traces, canonicalWorkloads] = await Promise.all([
    getSolutionsForDefinition(definition.name),
    getTracesForDefinition(definition.name),
    getCanonicalWorkloads(definition.type)
  ])
  
  return (
    <DefinitionPageContent
      definition={definition}
      solutions={solutions}
      traces={traces}
      canonicalWorkloads={canonicalWorkloads}
    />
  )
}