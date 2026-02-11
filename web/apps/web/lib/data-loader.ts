import { promises as fs } from "fs"
import path from "path"
import modelsData from "@/data/models"
import type { Definition, Solution, Trace, Model } from "./schemas"

// Get the flashinfer-trace path from environment or use default
const FLASHINFER_TRACE_PATH = process.env.FLASHINFER_TRACE_PATH || "/tmp/flashinfer-trace"
// Helper to resolve paths relative to the project root
function getDataPath(subPath: string): string {
  const basePath = path.resolve(FLASHINFER_TRACE_PATH)
  return path.join(basePath, subPath)
}

export async function getAllDefinitions(): Promise<Definition[]> {
  const definitionsDir = getDataPath("definitions")

  try {
    // Read all subdirectories (gemm, decode, prefill, etc.)
    const types = await fs.readdir(definitionsDir)
    const definitions: Definition[] = []

    for (const type of types) {
      const typePath = path.join(definitionsDir, type)
      const stat = await fs.stat(typePath)

      if (stat.isDirectory()) {
        const files = await fs.readdir(typePath)

        for (const file of files) {
          if (file.endsWith(".json")) {
            const content = await fs.readFile(path.join(typePath, file), "utf-8")
            try {
              const definition = JSON.parse(content) as Definition
              definitions.push(definition)
            } catch (e) {
              console.error(`Failed to parse definition ${file}:`, e)
            }
          }
        }
      }
    }

    return definitions.sort((a, b) => a.name.localeCompare(b.name))
  } catch (error) {
    console.error("Failed to load definitions:", error)
    return []
  }
}

export async function getDefinition(name: string): Promise<Definition | null> {
  const definitions = await getAllDefinitions()
  return definitions.find(d => d.name === name) || null
}

export async function getSolutionsForDefinition(definitionName: string): Promise<Solution[]> {
  const solutionsDir = getDataPath("solutions")

  try {
    // Try to read the solutions directory
    const exists = await fs.access(solutionsDir).then(() => true).catch(() => false)
    if (!exists) {
      console.log(`Solutions directory not found: ${solutionsDir}`)
      return []
    }

    // Read all author directories
    const authors = await fs.readdir(solutionsDir)
    const solutions: Solution[] = []

    for (const author of authors) {
      const authorPath = path.join(solutionsDir, author)
      const authorStat = await fs.stat(authorPath).catch(() => null)

      if (authorStat && authorStat.isDirectory()) {
        // Read all op_type directories under this author
        const types = await fs.readdir(authorPath)

        for (const type of types) {
          const typePath = path.join(authorPath, type)
          const typeStat = await fs.stat(typePath).catch(() => null)

          if (typeStat && typeStat.isDirectory()) {
            // Check if there's a subdirectory with the definition name
            const definitionPath = path.join(typePath, definitionName)
            const definitionStat = await fs.stat(definitionPath).catch(() => null)

            if (definitionStat && definitionStat.isDirectory()) {
              // Read solution files from the definition subdirectory
              const files = await fs.readdir(definitionPath)

              for (const file of files) {
                if (file.endsWith(".json")) {
                  const content = await fs.readFile(path.join(definitionPath, file), "utf-8")
                  try {
                    const solution = JSON.parse(content) as Solution
                    if (solution.definition === definitionName) {
                      solutions.push(solution)
                    }
                  } catch (e) {
                    console.error(`Failed to parse solution ${file}:`, e)
                  }
                }
              }
            }
          }
        }
      }
    }

    console.log(`Found ${solutions.length} solutions for ${definitionName}`)
    return solutions
  } catch (error) {
    console.error("Failed to load solutions:", error)
    return []
  }
}

export async function getTracesForDefinition(definitionName: string): Promise<Trace[]> {
  const tracesDir = getDataPath("traces")

  try {
    // Check if traces directory exists
    const exists = await fs.access(tracesDir).then(() => true).catch(() => false)
    if (!exists) {
      console.log(`Traces directory not found: ${tracesDir}`)
      return []
    }

    const traces: Trace[] = []

    // Read all subdirectories (gemm, gqa, mla, etc.)
    const types = await fs.readdir(tracesDir)

    for (const type of types) {
      if (type === "workload") {
        continue
      }
      const typePath = path.join(tracesDir, type)
      const stat = await fs.stat(typePath).catch(() => null)

      if (stat && stat.isDirectory()) {
        // Look for JSONL files in this directory
        const files = await fs.readdir(typePath)

        for (const file of files) {
          // Check if this file matches our definition name
          if (file === `${definitionName}.jsonl`) {
            const content = await fs.readFile(path.join(typePath, file), "utf-8")
            const lines = content.trim().split("\n")

            for (const line of lines) {
              if (line) {
                try {
                  const trace = JSON.parse(line) as Trace
                  if (trace.definition === definitionName) {
                    traces.push(trace)
                  }
                } catch (e) {
                  console.error(`Failed to parse trace line in ${file}:`, e)
                }
              }
            }
          }
        }
      }
    }

    console.log(`Found ${traces.length} traces for ${definitionName}`)
    return traces
  } catch (error) {
    console.error("Failed to load traces:", error)
    return []
  }
}

// Load models from local data (since these are UI-specific)
export async function getAllModels(): Promise<Model[]> {
  return modelsData
}

export async function getModel(id: string): Promise<Model | null> {
  return modelsData.find(model => model.id === id) ?? null
}
