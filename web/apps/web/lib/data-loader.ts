import { promises as fs } from "fs"
import path from "path"
import { Definition, Solution, Trace, CanonicalWorkload, Model } from "./schemas"

// Get the flashinfer-trace path from environment or use default
const FLASHINFER_TRACE_PATH = process.env.FLASHINFER_TRACE_PATH || "/tmp/flashinfer-trace"
const MODELS_DATA_PATH = process.env.MODELS_DATA_PATH || "./data/models"
const CANONICAL_WORKLOADS_PATH = process.env.CANONICAL_WORKLOADS_PATH || "./data/canonical-workloads"

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
    
    // Read all subdirectories (gemm, decode, prefill, etc.)
    const types = await fs.readdir(solutionsDir)
    const solutions: Solution[] = []
    
    for (const type of types) {
      const typePath = path.join(solutionsDir, type)
      const stat = await fs.stat(typePath).catch(() => null)
      
      if (stat && stat.isDirectory()) {
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
    
    const files = await fs.readdir(tracesDir)
    const traces: Trace[] = []
    
    for (const file of files) {
      if (file.endsWith(".jsonl")) {
        const content = await fs.readFile(path.join(tracesDir, file), "utf-8")
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
    
    console.log(`Found ${traces.length} traces for ${definitionName}`)
    return traces
  } catch (error) {
    console.error("Failed to load traces:", error)
    return []
  }
}

// Load models from local data (since these are UI-specific)
export async function getAllModels(): Promise<Model[]> {
  const modelsDir = path.join(process.cwd(), MODELS_DATA_PATH)
  
  try {
    const files = await fs.readdir(modelsDir)
    const models = await Promise.all(
      files
        .filter(file => file.endsWith(".json"))
        .map(async file => {
          const content = await fs.readFile(path.join(modelsDir, file), "utf-8")
          return JSON.parse(content) as Model
        })
    )
    
    return models
  } catch (error) {
    console.error("Failed to load models:", error)
    return []
  }
}

export async function getModel(id: string): Promise<Model | null> {
  try {
    const content = await fs.readFile(
      path.join(process.cwd(), MODELS_DATA_PATH, `${id}.json`),
      "utf-8"
    )
    return JSON.parse(content) as Model
  } catch {
    return null
  }
}

// Load canonical workloads from local data
export async function getCanonicalWorkloads(type: string): Promise<CanonicalWorkload[]> {
  try {
    const content = await fs.readFile(
      path.join(process.cwd(), CANONICAL_WORKLOADS_PATH, `${type}.json`),
      "utf-8"
    )
    return JSON.parse(content) as CanonicalWorkload[]
  } catch {
    return []
  }
}