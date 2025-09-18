"use client"

import { useState } from "react"
import {
  Card,
  CardContent,
  CardTitle,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@flashinfer-bench/ui"
import { Info } from "lucide-react"
import { Definition } from "@/lib/schemas"
import { cn } from "@flashinfer-bench/utils"

export function AxesSignatureSection({ definition }: { definition: Definition }) {
  const [hoveredAxis, setHoveredAxis] = useState<string | null>(null)
  const [hoveredTensor, setHoveredTensor] = useState<string | null>(null)

  return (
    <section id="tensors-axes">
      <div className="grid gap-4 md:grid-cols-[3fr_7fr]">
        {/* Left: Axes */}
        <div className="flex flex-col">
          <h2 className="text-2xl font-semibold mb-4">Axes</h2>
          <Card className="flex-1 flex flex-col h-[70vh]">
            <CardContent className="pt-6 flex-1">
              <div className="space-y-2 max-h-[70vh] overflow-auto">
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
                      {axis.type === "const" && "value" in axis
                        ? `${(axis as any).value}`
                        : `var`}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right: Tensors (Inputs + Outputs) */}
        <div className="flex flex-col">
          <h2 className="text-2xl font-semibold mb-4">Signature</h2>
          <Card className="flex-1 flex flex-col h-[70vh]">
            <CardContent className="p-0 flex-1">
              <div className="max-h-[70vh] overflow-auto">
                {/* Inputs sticky header */}
                <div className="sticky top-0 z-10 bg-card">
                  <div className="px-6 py-3 border-b">
                    <CardTitle className="text-lg">Inputs</CardTitle>
                  </div>
                </div>
                <div className="px-6 pt-4 pb-6">
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
                </div>

                {/* Outputs sticky header */}
                <div className="sticky top-0 z-10 bg-card">
                  <div className="px-6 py-3 border-y">
                    <CardTitle className="text-lg">Outputs</CardTitle>
                  </div>
                </div>
                <div className="px-6 pt-4 pb-6">
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
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
