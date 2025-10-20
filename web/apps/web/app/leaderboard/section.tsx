import { LeaderboardClient } from "./client"
import {
  computeFastPCurvesForAuthors,
  computeAuthorCorrectnessSummary,
  type BaselineConfig,
} from "@/lib/analytics"
import type { Solution, Trace, Definition } from "@/lib/schemas"
import type { CurvePoint } from "@/lib/analytics"

type LeaderboardEntry = {
  definition: Definition
  solutions: Solution[]
  traces: Trace[]
  baseline?: BaselineConfig
  baselineNames: string[]
}

type LeaderboardSectionProps = {
  entries: LeaderboardEntry[]
  baselineLabel: string
  initialPinnedP?: number
}

type DefinitionAuthorDetail = {
  definition: Definition
  curves: Record<string, CurvePoint[]>
  comparisonCounts: Record<string, number>
  totalComparisons: number
  solutionNamesByAuthor: Record<string, string[]>
}

export function LeaderboardSection({ entries, baselineLabel, initialPinnedP }: LeaderboardSectionProps) {
  const filteredEntries = entries.filter((entry) => entry.solutions.length > 0 && entry.traces.length > 0)

  const excludedAuthors = new Set<string>()
  for (const entry of entries) {
    const baselineNames = new Set(entry.baselineNames || [])
    if (baselineNames.size === 0) continue
    for (const solution of entry.solutions) {
      if (baselineNames.has(solution.name) && solution.author) {
        excludedAuthors.add(solution.author)
      }
    }
  }

  const fast = computeFastPCurvesForAuthors({
    datasets: filteredEntries.map((entry) => ({
      solutions: entry.solutions,
      traces: entry.traces,
      baseline: entry.baseline,
    })),
    sampleCount: 300,
  })

  const correctness = computeAuthorCorrectnessSummary({
    datasets: filteredEntries.map((entry) => ({
      solutions: entry.solutions,
      traces: entry.traces,
    })),
  })

  const definitionAuthorDetails: DefinitionAuthorDetail[] = filteredEntries.map((entry) => {
    const { curves, comparisonCounts, totalComparisons } = computeFastPCurvesForAuthors({
      datasets: [
        {
          solutions: entry.solutions,
          traces: entry.traces,
          baseline: entry.baseline,
        },
      ],
      sampleCount: 300,
    })

    const solutionNamesByAuthor = entry.solutions.reduce<Record<string, string[]>>((acc, solution) => {
      if (!solution.author) return acc
      const list = acc[solution.author] ?? []
      list.push(solution.name)
      acc[solution.author] = list
      return acc
    }, {})

    return {
      definition: entry.definition,
      curves,
      comparisonCounts,
      totalComparisons,
      solutionNamesByAuthor,
    }
  })

  return (
    <LeaderboardClient
      fast={fast}
      correctness={correctness}
      excludedAuthors={[...excludedAuthors]}
      baselineLabel={baselineLabel}
      initialPinnedP={initialPinnedP}
      definitionAuthorDetails={definitionAuthorDetails}
    />
  )
}
