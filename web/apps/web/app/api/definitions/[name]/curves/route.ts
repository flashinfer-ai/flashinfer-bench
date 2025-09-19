import { NextRequest, NextResponse } from "next/server"
import { getSolutionsForDefinition, getTracesForDefinition } from "@/lib/data-loader"
import { computeWinAtPCurves, WorkloadFilters, SolutionFilters, BaselineConfig } from "@/lib/analytics"
import baselinesData from "@/data/baselines.json"

function parseQueryArray(q: string | string[] | null): string[] | undefined {
  if (!q) return undefined
  const v = Array.isArray(q) ? q : q.split(",")
  return v.map((s) => s.trim()).filter(Boolean)
}

export async function GET(req: NextRequest, ctx: any) {
  try {
    const { searchParams } = new URL(req.url)
    const pMaybe = ctx?.params
    const p = pMaybe && typeof pMaybe.then === "function" ? await pMaybe : pMaybe
    const name = p?.name
    const solutions = await getSolutionsForDefinition(name)
    const traces = await getTracesForDefinition(name)

    // workload filters
    const onlyPassed = searchParams.get("onlyPassed") === "1"
    const devices = parseQueryArray(searchParams.get("devices"))
    const axisRangesParam = searchParams.get("axisRanges")
    let axisRanges: WorkloadFilters["axisRanges"] | undefined
    if (axisRangesParam) {
      try {
        axisRanges = JSON.parse(axisRangesParam)
      } catch {
        axisRanges = undefined
      }
    }

    const wf: WorkloadFilters = { onlyPassed, devices, axisRanges }

    // solution filters
    const languages = parseQueryArray(searchParams.get("languages"))
    const authors = parseQueryArray(searchParams.get("authors"))
    const targets = parseQueryArray(searchParams.get("targets"))
    const search = searchParams.get("search") || undefined
    const sf: SolutionFilters = { languages, authors, targets, search }

    const raw = Number(searchParams.get("sampleCount") || "201")
    const sampleCount = Math.max(51, Math.min(401, Number.isFinite(raw) ? raw : 201))

    const baselineConfig = (baselinesData as Record<string, Record<string, string> | undefined>)[name] || undefined
    const baseline: BaselineConfig | undefined = baselineConfig
      ? {
          default: baselineConfig.default,
          devices: Object.fromEntries(
            Object.entries(baselineConfig).filter(([key]) => key !== "default") as [string, string][]
          ),
        }
      : undefined

    const result = computeWinAtPCurves({
      traces,
      solutions,
      workloadFilters: wf,
      solutionFilters: sf,
      sampleCount,
      baseline,
    })
    return NextResponse.json(result)
  } catch (e: any) {
    console.error("curves endpoint error", e)
    return NextResponse.json({ error: e?.message || "unknown" }, { status: 500 })
  }
}
