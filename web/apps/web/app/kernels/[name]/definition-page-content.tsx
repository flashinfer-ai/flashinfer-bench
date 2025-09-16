"use client"

import { useState } from "react"
import { Definition, Solution, Trace, CanonicalWorkload } from "@/lib/schemas"
import { DefinitionReference } from "./reference"
import { TracesTable } from "./traces"
import { DefinitionHeader } from "./header"
import { AxesSignatureSection } from "./axes-sig"
import { ConstraintsSection } from "./constraints"
import { SolutionsSection } from "./solutions"

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
  const [hoveredAxis, setHoveredAxis] = useState<string | null>(null)
  return (
    <div className="relative">
      <DefinitionHeader
        definition={definition}
        solutionsCount={solutions.length}
        tracesCount={traces.length}
      />

      <div className="container py-8">
        <div className="space-y-8">
          <p className="text-muted-foreground">{definition.description}</p>
          <AxesSignatureSection
            definition={definition}
            hoveredAxis={hoveredAxis}
            setHoveredAxis={setHoveredAxis}
          />

          <ConstraintsSection definition={definition} />

          {/* Reference Implementation */}
          <section id="reference">
            <h2 className="text-2xl font-semibold mb-4">Reference Implementation</h2>
            <DefinitionReference definition={definition} />
          </section>

          <SolutionsSection definition={definition} solutions={solutions} />

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
