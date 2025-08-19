"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import { ArrowRight, Copy, Check, ChevronDown, ChevronRight, ChevronLeft, Search, Info } from "lucide-react"
import { Definition } from "@/lib/schemas"

interface DefinitionWithCounts extends Definition {
  solutionCount: number
  traceCount: number
}

interface KernelsSectionProps {
  definitions: DefinitionWithCounts[]
  initialSearch?: string
}

const ITEMS_PER_PAGE = 9

export function KernelsSection({ definitions, initialSearch = "" }: KernelsSectionProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [search, setSearch] = useState(initialSearch)
  const [selectedTab, setSelectedTab] = useState("all")
  const [currentPage, setCurrentPage] = useState(1)

  // Extract all unique scopes from definitions tags
  const scopes = Array.from(new Set(
    definitions.flatMap(d => 
      (d.tags || []).filter(tag => tag.startsWith('scope:')).map(tag => tag.replace('scope:', ''))
    )
  )).sort()

  // Filter definitions based on search
  const filteredDefinitions = definitions.filter(d => 
    d.name?.toLowerCase().includes(search.toLowerCase()) ||
    d.type?.toLowerCase().includes(search.toLowerCase()) ||
    (d.tags || []).some(tag => tag.toLowerCase().includes(search.toLowerCase()))
  )

  // Group definitions by scope
  const scopeGroups: Record<string, DefinitionWithCounts[]> = {
    all: filteredDefinitions
  }
  
  // Create groups for each scope
  scopes.forEach(scope => {
    scopeGroups[scope] = filteredDefinitions.filter(d => 
      (d.tags || []).includes(`scope:${scope}`)
    )
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
    // Extract type from tags (type:{type})
    const typeTag = (def.tags || []).find(tag => tag.startsWith('type:'))
    const type = typeTag ? typeTag.replace('type:', '') : (def.type || 'unknown')
    
    const typeColor = {
      gemm: "bg-blue-500",
      decode: "bg-green-500",
      prefill: "bg-purple-500",
      attention: "bg-indigo-500"
    }[type] || "bg-gray-500"

    // Extract tags for display
    const tags = def.tags || []
    const modelTags = tags.filter(tag => tag.startsWith('model:'))
    const statusTags = tags.filter(tag => tag.startsWith('status:'))
    const scopeTags = tags.filter(tag => tag.startsWith('scope:'))
    const typeTags = tags.filter(tag => tag.startsWith('type:'))

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
                  {typeTags.map(tag => (
                    <span key={tag} className={`text-xs px-2 py-1 rounded-full ${
                      type === "gemm" ? "bg-blue-100 text-blue-700" :
                      type === "decode" ? "bg-green-100 text-green-700" :
                      type === "prefill" ? "bg-purple-100 text-purple-700" :
                      type === "attention" ? "bg-indigo-100 text-indigo-700" :
                      "bg-gray-100 text-gray-700"
                    }`}>
                      {tag.replace('type:', '')}
                    </span>
                  ))}
                  {scopeTags.map(tag => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag.replace('scope:', '')}
                    </Badge>
                  ))}
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
                              <span className="font-mono">{name}</span>: {input.dtype} [{input.shape.join(", ")}]
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium mb-2">Outputs</p>
                        <div className="space-y-1">
                          {Object.entries(def.outputs).map(([name, output]) => (
                            <div key={name} className="text-sm text-muted-foreground">
                              <span className="font-mono">{name}</span>: {output.dtype} [{output.shape.join(", ")}]
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
        <TabsList>
          <TabsTrigger value="all">All ({scopeGroups.all.length})</TabsTrigger>
          {scopes.map(scope => (
            <TabsTrigger key={scope} value={scope}>
              {scope.charAt(0).toUpperCase() + scope.slice(1)} ({scopeGroups[scope]?.length || 0})
            </TabsTrigger>
          ))}
        </TabsList>
        
        <TabsContent value="all" className="mt-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {scopeGroups.all
              .slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE)
              .map(renderDefinitionCard)}
          </div>
          {scopeGroups.all.length === 0 && (
            <p className="text-center text-muted-foreground py-8">No kernels found matching &quot;{search}&quot;</p>
          )}
          {renderPagination(scopeGroups.all.length)}
        </TabsContent>
        
        {scopes.map(scope => (
          <TabsContent key={scope} value={scope} className="mt-6">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {(scopeGroups[scope] || [])
                .slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE)
                .map(renderDefinitionCard)}
            </div>
            {(scopeGroups[scope] || []).length === 0 && (
              <p className="text-center text-muted-foreground py-8">No {scope} kernels found</p>
            )}
            {renderPagination((scopeGroups[scope] || []).length)}
          </TabsContent>
        ))}
      </Tabs>
    </section>
  )
}