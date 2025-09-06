"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import { Copy, Check, ChevronRight, ChevronLeft, Search, Info } from "lucide-react"
import { Definition } from "@/lib/schemas"

interface DefinitionWithCounts extends Definition {
  solutionCount: number
  traceCount: number
}

interface KernelsSectionProps {
  definitions: DefinitionWithCounts[]
}

const ITEMS_PER_PAGE = 9

export function KernelsSection({ definitions }: KernelsSectionProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [search, setSearch] = useState("")
  const [selectedTab, setSelectedTab] = useState("all")
  const [currentPage, setCurrentPage] = useState(1)

  // Extract all unique types from definitions
  const types = Array.from(new Set(
    definitions.map(d => d.type).filter(Boolean)
  )).sort()

  // Filter definitions based on search
  const filteredDefinitions = definitions.filter(d =>
    d.name?.toLowerCase().includes(search.toLowerCase()) ||
    d.type?.toLowerCase().includes(search.toLowerCase()) ||
    (d.tags || []).some(tag => tag.toLowerCase().includes(search.toLowerCase()))
  )

  // Group definitions by type
  const typeGroups: Record<string, DefinitionWithCounts[]> = {
    all: filteredDefinitions
  }

  // Create groups for each type
  types.forEach(type => {
    typeGroups[type] = filteredDefinitions.filter(d => d.type === type)
  })

  // Reset page when search or tab changes
  const handleSearch = (value: string) => {
    setSearch(value)
    setCurrentPage(1)
  }

  const handleTabChange = (value: string) => {
    setSelectedTab(value)
    setCurrentPage(1)
  }

  const copyToClipboard = async (text: string, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    try {
      await navigator.clipboard.writeText(text)
      setCopiedId(text)
      setTimeout(() => setCopiedId(null), 2000)
    } catch (err) {
      console.error("Failed to copy:", err)
    }
  }


  const renderDefinitionCard = (def: DefinitionWithCounts) => {
    const type = def.type || 'unknown'

    const typeColor = {
      // Attention types - green shades
      mha: "bg-green-500",
      gqa: "bg-emerald-500",
      mla: "bg-teal-500",
      // GEMM types - blue shades
      gemm: "bg-blue-500",
      grouped_gemm: "bg-sky-500",
      batch_gemm: "bg-indigo-500",
      // Misc types - purple/orange shades
      sampling: "bg-purple-500",
      norm: "bg-orange-500",
      moe: "bg-pink-500"
    }[type] || "bg-gray-500"

    // Extract tags for display
    const tags = def.tags || []
    const modelTags = tags.filter(tag => tag.startsWith('model:'))
    const statusTags = tags.filter(tag => tag.startsWith('status:'))

    return (
      <Link key={def.name} href={`/kernels/${def.name}`}>
        <div className="relative">
          <div className={`absolute left-0 top-0 bottom-0 w-1 ${typeColor} rounded-l`} />
          <Card className="h-full ml-1 hover:shadow-lg hover:border-primary transition-all cursor-pointer">
            <CardHeader className="pb-3">
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="group flex items-center gap-2">
                      <CardTitle className="text-base font-mono truncate">{def.name}</CardTitle>
                      <button
                        onClick={(e) => copyToClipboard(def.name, e)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity"
                        aria-label="Copy definition name"
                      >
                        {copiedId === def.name ? (
                          <Check className="h-3 w-3 text-green-600" />
                        ) : (
                          <Copy className="h-3 w-3 text-muted-foreground hover:text-foreground" />
                        )}
                      </button>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {def.solutionCount} solutions • {def.traceCount} traces
                    </p>
                  </div>
                </div>

                {/* Display tags */}
                <div className="flex flex-wrap gap-1">
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    // Attention types
                    type === "mha" ? "bg-green-100 text-green-700" :
                    type === "gqa" ? "bg-emerald-100 text-emerald-700" :
                    type === "mla" ? "bg-teal-100 text-teal-700" :
                    // GEMM types
                    type === "gemm" ? "bg-blue-100 text-blue-700" :
                    type === "grouped_gemm" ? "bg-sky-100 text-sky-700" :
                    type === "batch_gemm" ? "bg-indigo-100 text-indigo-700" :
                    // Misc types
                    type === "sampling" ? "bg-purple-100 text-purple-700" :
                    type === "norm" ? "bg-orange-100 text-orange-700" :
                    type === "moe" ? "bg-pink-100 text-pink-700" :
                    "bg-gray-100 text-gray-700"
                  }`}>
                    {type.replace('_', ' ').toUpperCase()}
                  </span>
                  {modelTags.map(tag => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag.replace('model:', '')}
                    </Badge>
                  ))}
                  {statusTags.map(tag => (
                    <Badge key={tag} variant={tag.includes('draft') ? "destructive" : "default"} className="text-xs">
                      {tag.replace('status:', '')}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="flex items-center gap-1 text-sm text-muted-foreground">
                <HoverCard>
                  <HoverCardTrigger asChild>
                    <button
                      onClick={(e) => e.stopPropagation()}
                      className="flex items-center gap-1 hover:text-foreground transition-colors"
                    >
                      <Info className="h-3 w-3" />
                      <span className="font-medium">Axes</span>
                      <span className="text-xs">({Object.keys(def.axes).length})</span>
                      <span className="ml-2 font-medium">Inputs</span>
                      <span className="text-xs">({Object.keys(def.inputs).length})</span>
                      <span className="ml-2 font-medium">Outputs</span>
                      <span className="text-xs">({Object.keys(def.outputs).length})</span>
                    </button>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-96" align="start">
                    <div className="space-y-3">
                      {def.description && (
                        <div>
                          <p className="text-sm text-muted-foreground">{def.description}</p>
                        </div>
                      )}
                      <div>
                        <p className="text-sm font-medium mb-2">Axes</p>
                        <div className="space-y-1">
                          {Object.entries(def.axes).map(([name, axis]) => (
                            <div key={name} className="text-sm text-muted-foreground">
                              <span className="font-mono">{name}</span>: {
                                axis.type === "const" && 'value' in axis
                                  ? `const = ${axis.value}`
                                  : `variable${axis.parent ? ` (parent: ${axis.parent})` : ''}`
                              }
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium mb-2">Inputs</p>
                        <div className="space-y-1">
                          {Object.entries(def.inputs).map(([name, input]) => (
                            <div key={name} className="text-sm text-muted-foreground">
                              <span className="font-mono">{name}</span>: {input.dtype} {input.shape ? `[${input.shape.join(", ")}]` : "Scalar"}
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium mb-2">Outputs</p>
                        <div className="space-y-1">
                          {Object.entries(def.outputs).map(([name, output]) => (
                            <div key={name} className="text-sm text-muted-foreground">
                              <span className="font-mono">{name}</span>: {output.dtype} {output.shape ? `[${output.shape.join(", ")}]` : "Scalar"}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </HoverCardContent>
                </HoverCard>
              </div>
            </CardContent>
          </Card>
        </div>
      </Link>
    )
  }

  const renderPagination = (totalItems: number) => {
    const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE)

    if (totalPages <= 1) return null

    return (
      <div className="flex items-center justify-center gap-2 mt-6">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
          disabled={currentPage === 1}
        >
          <ChevronLeft className="h-4 w-4" />
          Previous
        </Button>
        <div className="flex items-center gap-1">
          {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
            <Button
              key={page}
              variant={currentPage === page ? "default" : "outline"}
              size="sm"
              onClick={() => setCurrentPage(page)}
              className="w-10"
            >
              {page}
            </Button>
          ))}
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
          disabled={currentPage === totalPages}
        >
          Next
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  return (
    <section className="container space-y-6 py-8 md:py-12">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h2 className="text-3xl font-bold tracking-tight">Kernels</h2>
            <p className="text-muted-foreground">
              Browse kernel specifications and their traces
            </p>
          </div>
        </div>

        {/* Search bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search kernels..."
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            className="pl-10"
          />
        </div>
      </div>

      <Tabs value={selectedTab} onValueChange={handleTabChange} className="w-full">
        <TabsList className="flex-wrap h-auto">
          <TabsTrigger value="all">All ({typeGroups.all.length})</TabsTrigger>
          {types.map(type => (
            <TabsTrigger key={type} value={type}>
              {type.replace(/_/g, ' ').toUpperCase()} ({typeGroups[type]?.length || 0})
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="all" className="mt-6">
          {definitions.length === 0 ? (
            <p className="text-center text-muted-foreground py-8">No kernels found</p>
          ) : (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {typeGroups.all
                  .slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE)
                  .map(renderDefinitionCard)}
              </div>
              {typeGroups.all.length === 0 && search && (
                <p className="text-center text-muted-foreground py-8">No kernels found matching &quot;{search}&quot;</p>
              )}
              {renderPagination(typeGroups.all.length)}
            </>
          )}
        </TabsContent>

        {types.map(type => (
          <TabsContent key={type} value={type} className="mt-6">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {(typeGroups[type] || [])
                .slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE)
                .map(renderDefinitionCard)}
            </div>
            {(typeGroups[type] || []).length === 0 && (
              <p className="text-center text-muted-foreground py-8">No {type} kernels found</p>
            )}
            {renderPagination((typeGroups[type] || []).length)}
          </TabsContent>
        ))}
      </Tabs>
    </section>
  )
}
