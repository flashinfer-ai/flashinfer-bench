"use client"

import { Button } from "@flashinfer-bench/ui"
import { CheckSquare, Filter, X } from "lucide-react"

export type ToolbarChip = {
  label: string
  onRemove?: () => void
}

export type SolutionsTracesToolbarProps = {
  onOpenWorkload: () => void
  onOpenSolution: () => void
  chips: ToolbarChip[]
  counts: { solutions: number; workloads: number }
}

function Chip({ label, onRemove }: ToolbarChip) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-1 text-xs">
      {label}
      {onRemove && (
        <button className="text-muted-foreground hover:text-foreground" onClick={onRemove}>
          <X className="h-3 w-3" />
        </button>
      )}
    </span>
  )
}

export function SolutionsTracesToolbar({
  onOpenWorkload,
  onOpenSolution,
  chips,
  counts,
}: SolutionsTracesToolbarProps) {
  return (
    <div className="sticky top-14 z-30 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
      <div className="container py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2">
            {chips.map((chip, index) => (
              <Chip key={`${chip.label}-${index}`} label={chip.label} onRemove={chip.onRemove} />
            ))}
          </div>
          <Button variant="ghost" size="sm" onClick={onOpenWorkload}>
            <Filter className="h-4 w-4 mr-1" />Edit Workload Filters
          </Button>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <span>Solutions: {counts.solutions}</span>
          <span>Workloads: {counts.workloads}</span>
          <Button variant="ghost" size="sm" onClick={onOpenSolution}>
            <CheckSquare className="h-4 w-4 mr-1" />Solution Filters
          </Button>
        </div>
      </div>
    </div>
  )
}
