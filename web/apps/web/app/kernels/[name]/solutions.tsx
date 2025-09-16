"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle, Badge, Button } from "@flashinfer-bench/ui"
import { Code2 } from "lucide-react"
import { Definition, Solution } from "@/lib/schemas"

export function SolutionsSection({
  definition,
  solutions,
}: {
  definition: Definition
  solutions: Solution[]
}) {
  return (
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
  )
}
