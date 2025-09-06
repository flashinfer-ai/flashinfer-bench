"use client"

import { useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import { Copy, Check, ArrowLeft, ExternalLink, Info, Code2 } from "lucide-react"
import { Definition, Solution, Trace, CanonicalWorkload } from "@/lib/schemas"
import { DefinitionDetails } from "./definition-details"
import { TracesTable } from "./traces-table"
import { cn } from "@/lib/utils"

interface DefinitionPageContentProps {
  definition: Definition
  solutions: Solution[]
  traces: Trace[]
  canonicalWorkloads: CanonicalWorkload[]
}

export function DefinitionPageContent({
  definition,
  solutions,
  traces,
  canonicalWorkloads
}: DefinitionPageContentProps) {
  const [copiedItem, setCopiedItem] = useState<string | null>(null)
  const [hoveredAxis, setHoveredAxis] = useState<string | null>(null)
  const [hoveredTensor, setHoveredTensor] = useState<string | null>(null)

  const copyToClipboard = async (text: string, type: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedItem(type)
      setTimeout(() => setCopiedItem(null), 2000)
    } catch (err) {
      console.error("Failed to copy:", err)
    }
  }

  const copyJSON = () => {
    // Preserve the exact order of fields as they appear in the original definition
    const orderedDefinition: any = {}

    // Copy all fields in their original order
    Object.keys(definition).forEach(key => {
      orderedDefinition[key] = definition[key as keyof Definition]
    })

    copyToClipboard(JSON.stringify(orderedDefinition, null, 2), "json")
  }


  return (
    <div className="relative">
      {/* Sticky Header */}
      <div className="sticky top-14 z-40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
        <div className="container py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">
                <ArrowLeft className="h-4 w-4" />
              </Link>
              <h1 className="text-xl font-mono font-bold">{definition.name}</h1>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <Button
                variant="ghost"
                size="sm"
                onClick={copyJSON}
              >
                {copiedItem === "json" ? (
                  <>
                    <Check className="h-3 w-3 mr-1" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3 mr-1" />
                    Copy JSON
                  </>
                )}
              </Button>
              <span className="text-muted-foreground">·</span>
              <a href="#solutions" className="hover:underline">
                Solutions ({solutions.length})
              </a>
              <span className="text-muted-foreground">·</span>
              <a href="#traces" className="hover:underline">
                Traces ({traces.length})
              </a>
            </div>
          </div>
        </div>
      </div>

      <div className="container py-8">
        <div className="space-y-8">
          <p className="text-muted-foreground">{definition.description}</p>

          {/* Axes Section */}
          <section id="axes">
            <h2 className="text-2xl font-semibold mb-4">Axes</h2>
            <Card>
              <CardContent className="pt-6">
                <div className="space-y-2">
                  {Object.entries(definition.axes).map(([name, axis]) => (
                    <div
                      key={name}
                      className={cn(
                        "flex items-center justify-between p-2 rounded-md transition-colors",
                        hoveredAxis === name && "bg-muted"
                      )}
                      onMouseEnter={() => setHoveredAxis(name)}
                      onMouseLeave={() => setHoveredAxis(null)}
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-medium">{name}</span>
                        {axis.description && (
                          <HoverCard>
                            <HoverCardTrigger asChild>
                              <Info className="h-3 w-3 text-muted-foreground" />
                            </HoverCardTrigger>
                            <HoverCardContent className="w-80">
                              <p className="text-sm">{axis.description}</p>
                            </HoverCardContent>
                          </HoverCard>
                        )}
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {axis.type === "const" && 'value' in axis
                          ? `const = ${axis.value}`
                          : `variable${axis.parent ? ` (parent: ${axis.parent})` : ''}`}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </section>

          {/* Tensors Section */}
          <section id="tensors">
            <h2 className="text-2xl font-semibold mb-4">Tensors</h2>
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Inputs</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Shape</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {Object.entries(definition.inputs).map(([name, tensor]) => (
                        <TableRow
                          key={name}
                          className={cn(
                            "transition-colors",
                            hoveredTensor === name && "bg-muted"
                          )}
                          onMouseEnter={() => setHoveredTensor(name)}
                          onMouseLeave={() => setHoveredTensor(null)}
                        >
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <span className="font-mono">{name}</span>
                              {tensor.description && (
                                <HoverCard>
                                  <HoverCardTrigger asChild>
                                    <Info className="h-3 w-3 text-muted-foreground" />
                                  </HoverCardTrigger>
                                  <HoverCardContent className="w-80">
                                    <p className="text-sm">{tensor.description}</p>
                                  </HoverCardContent>
                                </HoverCard>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>{tensor.dtype}</TableCell>
                          <TableCell>
                            <span className="font-mono text-sm">
                              {tensor.shape ? (
                                <>
                                  [{tensor.shape.map((s, i) => (
                                    <span key={i}>
                                      {i > 0 && ", "}
                                      <span className={cn(
                                        hoveredAxis === s && "text-primary font-semibold"
                                      )}>
                                        {s}
                                      </span>
                                    </span>
                                  ))}]
                                </>
                              ) : (
                                "Scalar"
                              )}
                            </span>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Outputs</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Shape</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {Object.entries(definition.outputs).map(([name, tensor]) => (
                        <TableRow
                          key={name}
                          className={cn(
                            "transition-colors",
                            hoveredTensor === name && "bg-muted"
                          )}
                          onMouseEnter={() => setHoveredTensor(name)}
                          onMouseLeave={() => setHoveredTensor(null)}
                        >
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <span className="font-mono">{name}</span>
                              {tensor.description && (
                                <HoverCard>
                                  <HoverCardTrigger asChild>
                                    <Info className="h-3 w-3 text-muted-foreground" />
                                  </HoverCardTrigger>
                                  <HoverCardContent className="w-80">
                                    <p className="text-sm">{tensor.description}</p>
                                  </HoverCardContent>
                                </HoverCard>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>{tensor.dtype}</TableCell>
                          <TableCell>
                            <span className="font-mono text-sm">
                              {tensor.shape ? (
                                <>
                                  [{tensor.shape.map((s, i) => (
                                    <span key={i}>
                                      {i > 0 && ", "}
                                      <span className={cn(
                                        hoveredAxis === s && "text-primary font-semibold"
                                      )}>
                                        {s}
                                      </span>
                                    </span>
                                  ))}]
                                </>
                              ) : (
                                "scalar"
                              )}
                            </span>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </div>
          </section>

          {/* Constraints Section */}
          {definition.constraints && definition.constraints.length > 0 && (
            <section id="constraints">
              <h2 className="text-2xl font-semibold mb-4">Constraints</h2>
              <Card>
                <CardContent className="pt-6">
                  <ul className="space-y-2">
                    {definition.constraints.map((constraint, idx) => (
                      <li key={idx} className="text-sm font-mono text-muted-foreground">
                        • {constraint}
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </section>
          )}

          {/* Reference Implementation */}
          <section id="reference">
            <h2 className="text-2xl font-semibold mb-4">Reference Implementation</h2>
            <DefinitionDetails definition={definition} />
          </section>

          {/* Solutions Section */}
          <section id="solutions">
            <h2 className="text-2xl font-semibold mb-4">Solutions</h2>
            <div className="grid gap-4">
              {solutions.length > 0 ? (
                solutions.map((solution) => (
                  <Card key={solution.name}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <CardTitle className="text-lg font-mono">{solution.name}</CardTitle>
                          <CardDescription>{solution.description}</CardDescription>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary">{solution.author}</Badge>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              // Store solution in sessionStorage and pass ID via URL
                              const solutionId = `${definition.name}-${solution.name}`.replace(/[^a-zA-Z0-9-_]/g, '_')
                              sessionStorage.setItem(`solution-${solutionId}`, JSON.stringify(solution))
                              window.open(`/editor?solution=${solutionId}`, '_blank')
                            }}
                          >
                            <Code2 className="h-4 w-4 mr-1" />
                            Open in Editor
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <span>Language: {solution.spec.language}</span>
                        <span>•</span>
                        <span>Targets: {solution.spec.target_hardware.join(", ")}</span>
                      </div>
                    </CardContent>
                  </Card>
                ))
              ) : (
                <Card>
                  <CardContent className="py-8 text-center text-muted-foreground">
                    No solutions available yet
                  </CardContent>
                </Card>
              )}
            </div>
          </section>

          {/* Traces Section */}
          <section id="traces">
            <h2 className="text-2xl font-semibold mb-4">Traces</h2>
            <TracesTable
              traces={traces}
              solutions={solutions}
              canonicalWorkloads={canonicalWorkloads}
            />
          </section>
        </div>
      </div>
    </div>
  )
}
