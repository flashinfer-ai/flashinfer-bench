"use client"

import Link from "next/link"
import { useState, useEffect, useCallback } from "react"
import { Editor as MonacoEditor } from "@monaco-editor/react"
import { Badge, Button, Card, toast } from "@flashinfer-bench/ui"
import { ArrowLeft, Copy, Download, Check, Plus, FileText, Code } from "lucide-react"
import type { Axis, Definition, Solution, Tensor, Trace, WorkloadInput } from "@/lib/schemas"

interface ViewerProps {
  data: any
  onBack: () => void
}

export function Viewer({ data, onBack }: ViewerProps) {
  const isTrace = data && typeof data === "object" && "workload" in data

  if (isTrace) {
    return <TraceViewer data={data} onBack={onBack} />
  }

  return <DefinitionSolutionViewer data={data} onBack={onBack} />
}

function DefinitionSolutionViewer({ data, onBack }: ViewerProps) {
  const [jsonText, setJsonText] = useState("")
  const [referenceCode, setReferenceCode] = useState("")
  const [sourceCode, setSourceCode] = useState<Record<string, string>>({})
  const [activeSourceFile, setActiveSourceFile] = useState<string>("")
  const [copied, setCopied] = useState(false)
  const [jsonError, setJsonError] = useState<string | null>(null)

  const isDefinition = data.reference !== undefined
  const isSolution = data.sources !== undefined

  // Initialize state from data - run only once
  useEffect(() => {
    if (isDefinition) {
      setReferenceCode(data.reference || "")
    } else if (isSolution && data.sources) {
      const codes: Record<string, string> = {}
      data.sources.forEach((file: any) => {
        codes[file.path] = file.content || ""
      })
      setSourceCode(codes)
      if (data.sources.length > 0) {
        setActiveSourceFile(data.sources[0].path)
      }
    }

    // Initialize JSON preview
    setJsonText(JSON.stringify(data, null, 2))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])

  // Update JSON preview with current code
  const updateJsonPreview = useCallback((baseData?: any) => {
    const dataToUse = baseData || data
    let jsonData = { ...dataToUse }

    if (isDefinition) {
      jsonData.reference = referenceCode || dataToUse.reference || ""
    } else if (isSolution) {
      const sources = Object.entries(sourceCode).map(([path, content]) => ({
        path,
        content
      }))
      jsonData.sources = sources.length > 0 ? sources : dataToUse.sources || []
    }

    setJsonText(JSON.stringify(jsonData, null, 2))
  }, [data, isDefinition, isSolution, referenceCode, sourceCode])

  // Update JSON preview when code changes
  useEffect(() => {
    if (!data) return
    updateJsonPreview()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [referenceCode, sourceCode])

  // Parse current JSON
  const getParsedJson = () => {
    try {
      return JSON.parse(jsonText)
    } catch (e) {
      return null
    }
  }

  // Validate JSON only
  useEffect(() => {
    try {
      JSON.parse(jsonText)
      setJsonError(null)
    } catch (e) {
      setJsonError(e instanceof Error ? e.message : "Invalid JSON")
    }
  }, [jsonText])

  // Smart insertion functions
  const insertAxis = () => {
    const data = getParsedJson()
    if (!data) {
      toast({ description: "Please fix JSON errors first.", variant: "destructive" })
      return
    }

    if (!data.axes) data.axes = {}
    data.axes[`new_axis_${Object.keys(data.axes || {}).length + 1}`] = {
      type: "const",
      value: 1
    }
    setJsonText(JSON.stringify(data, null, 2))
  }

  const insertInput = () => {
    const data = getParsedJson()
    if (!data) {
      toast({ description: "Please fix JSON errors first.", variant: "destructive" })
      return
    }

    if (!data.inputs) data.inputs = {}
    data.inputs[`new_input_${Object.keys(data.inputs || {}).length + 1}`] = {
      shape: ["M", "N"],
      dtype: "float16"
    }
    setJsonText(JSON.stringify(data, null, 2))
  }

  const insertOutput = () => {
    const data = getParsedJson()
    if (!data) {
      toast({ description: "Please fix JSON errors first.", variant: "destructive" })
      return
    }

    if (!data.outputs) data.outputs = {}
    data.outputs[`new_output_${Object.keys(data.outputs || {}).length + 1}`] = {
      shape: ["M", "N"],
      dtype: "float16"
    }
    setJsonText(JSON.stringify(data, null, 2))
  }

  const insertConstraint = () => {
    const data = getParsedJson()
    if (!data) {
      toast({ description: "Please fix JSON errors first.", variant: "destructive" })
      return
    }

    if (!data.constraints) data.constraints = []
    data.constraints.push("// Add constraint expression here")
    setJsonText(JSON.stringify(data, null, 2))
  }

  const insertDependency = () => {
    const data = getParsedJson()
    if (!data) {
      toast({ description: "Please fix JSON errors first.", variant: "destructive" })
      return
    }

    if (!data.spec) data.spec = {}
    if (!data.spec.dependencies) data.spec.dependencies = []
    data.spec.dependencies.push("new-dependency")
    setJsonText(JSON.stringify(data, null, 2))
  }

  const insertTargetHardware = () => {
    const data = getParsedJson()
    if (!data) {
      toast({ description: "Please fix JSON errors first.", variant: "destructive" })
      return
    }

    if (!data.spec) data.spec = {}
    if (!data.spec.target_hardware) data.spec.target_hardware = []
    data.spec.target_hardware.push("cuda:90")
    setJsonText(JSON.stringify(data, null, 2))
  }

  const exportToJson = () => {
    const mergedData = getParsedJson()
    if (!mergedData) {
      toast({ description: "Please fix JSON errors before exporting", variant: "destructive" })
      return
    }

    try {
      const jsonString = JSON.stringify(mergedData, null, 2)
      const blob = new Blob([jsonString], { type: "application/json" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${mergedData.name || (isDefinition ? "definition" : "solution")}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      toast({ description: "JSON exported successfully!" })
    } catch (error) {
      toast({ description: "Failed to export JSON", variant: "destructive" })
    }
  }

  const copyToClipboard = async () => {
    const mergedData = getParsedJson()
    if (!mergedData) {
      toast({ description: "Please fix JSON errors before copying", variant: "destructive" })
      return
    }

    try {
      const jsonString = JSON.stringify(mergedData, null, 2)
      await navigator.clipboard.writeText(jsonString)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
      toast({ description: "Copied to clipboard!" })
    } catch (error) {
      toast({ description: "Failed to copy to clipboard", variant: "destructive" })
    }
  }

  const definitionData = isDefinition ? (data as Definition) : null
  const axesEntries = definitionData ? Object.entries(definitionData.axes || {}) : []
  const inputEntries = definitionData ? Object.entries(definitionData.inputs || {}) : []
  const outputEntries = definitionData ? Object.entries(definitionData.outputs || {}) : []
  const constraintEntries = definitionData?.constraints ?? []
  const tagList = definitionData?.tags ?? []
  const solutionData = isSolution ? (data as Solution) : null
  const spec = solutionData?.spec ?? null
  const targetHardware = spec?.target_hardware ?? []
  const dependencies = spec?.dependencies ?? []
  const buildCommands = spec?.build_commands ?? []
  const sourceCount = Object.keys(sourceCode).length

  const formatAxisSummary = (axis: Axis) => {
    if (axis.type === "const") {
      return `const = ${axis.value}`
    }
    const suffix = axis.parent ? ` (parent: ${axis.parent})` : ""
    return `variable${suffix}`
  }

  const formatTensorShape = (tensor: Tensor) => {
    if (!tensor.shape || tensor.shape.length === 0) return "Unspecified"
    return tensor.shape.join(" × ")
  }

  const languageForFile = (path: string) => {
    const lower = path.toLowerCase()
    if (lower.endsWith(".ts") || lower.endsWith(".tsx")) return "typescript"
    if (lower.endsWith(".js") || lower.endsWith(".jsx")) return "javascript"
    if (lower.endsWith(".py")) return "python"
    if (lower.endsWith(".rs")) return "rust"
    if (lower.endsWith(".sh")) return "shell"
    if (lower.endsWith(".md")) return "markdown"
    if (lower.endsWith(".json")) return "json"
    if (lower.endsWith(".yaml") || lower.endsWith(".yml")) return "yaml"
    if (lower.endsWith(".toml")) return "toml"
    if (lower.endsWith(".sql")) return "sql"
    if (
      lower.endsWith(".cu") ||
      lower.endsWith(".cuh") ||
      lower.endsWith(".cpp") ||
      lower.endsWith(".cc") ||
      lower.endsWith(".hpp") ||
      lower.endsWith(".h") ||
      lower.endsWith(".c")
    )
      return "cpp"
    return "plaintext"
  }

  const codePanel = (
    <Card className="p-4 flex flex-col h-full">
      <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
        <Code className="h-5 w-5" />
        {isDefinition ? "Reference Implementation" : "Source Code"}
      </h3>

      <div className="flex-1 flex flex-col min-h-0">
        {isDefinition ? (
          <div className="border rounded-lg overflow-hidden h-full">
            <MonacoEditor
              height="100%"
              defaultLanguage="python"
              value={referenceCode}
              onChange={(value) => setReferenceCode(value || "")}
              theme="vs-dark"
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                lineNumbers: "on",
                wordWrap: "on",
                automaticLayout: true,
              }}
            />
          </div>
        ) : isSolution ? (
          activeSourceFile ? (
            <div className="border rounded-lg overflow-hidden h-full">
              <MonacoEditor
                height="100%"
                language={languageForFile(activeSourceFile)}
                value={sourceCode[activeSourceFile] || ""}
                onChange={(value) =>
                  setSourceCode((prev) => ({
                    ...prev,
                    [activeSourceFile]: value || "",
                  }))
                }
                theme="vs-dark"
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  lineNumbers: "on",
                  wordWrap: "on",
                  automaticLayout: true,
                }}
              />
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-muted-foreground">
              Select a source file to view its contents.
            </div>
          )
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            No code content found.
          </div>
        )}
      </div>
    </Card>
  )

  const definitionPanel = isDefinition && definitionData ? (
    <Card className="p-6 flex h-full flex-col space-y-4 overflow-hidden">
      <h3 className="text-lg font-semibold">Definition Summary</h3>

      <div className="space-y-3 text-sm text-muted-foreground break-words">
        <div>
          <span className="text-foreground font-medium">Name:</span>{" "}
          <span className="font-mono text-foreground">{definitionData.name}</span>
        </div>
        <div>
          <span className="text-foreground font-medium">Op Type:</span>{" "}
          <span className="uppercase tracking-wide text-muted-foreground">{definitionData.op_type}</span>
        </div>
        {definitionData.description && (
          <div>
            <span className="text-foreground font-medium">Description:</span>
            <p className="mt-1 whitespace-pre-wrap">{definitionData.description}</p>
          </div>
        )}
        {tagList.length > 0 && (
          <div>
            <span className="text-foreground font-medium">Tags:</span>
            <div className="mt-2 flex flex-wrap gap-2">
              {tagList.map((tag) => (
                <Badge key={tag} variant="outline">
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-auto pr-2 text-sm text-muted-foreground space-y-4">
        <div>
          <p className="text-foreground font-medium">Axes</p>
          {axesEntries.length ? (
            <ul className="ml-4 list-disc space-y-1">
              {axesEntries.map(([name, axis]) => {
                const axisDetails = axis as Axis
                return (
                  <li key={name}>
                    <span className="font-mono text-foreground">{name}</span>: {formatAxisSummary(axisDetails)}
                    {axisDetails.description ? (
                      <span className="block ml-4 text-xs text-muted-foreground">{axisDetails.description}</span>
                    ) : null}
                  </li>
                )
              })}
            </ul>
          ) : (
            <p>No axes specified.</p>
          )}
        </div>

        <div>
          <p className="text-foreground font-medium">Inputs</p>
          {inputEntries.length ? (
            <ul className="ml-4 list-disc space-y-2">
              {inputEntries.map(([name, tensor]) => {
                const tensorDetails = tensor as Tensor
                return (
                  <li key={name}>
                    <span className="font-mono text-foreground">{name}</span>: {tensorDetails.dtype}
                    <span className="ml-2 text-muted-foreground">[{formatTensorShape(tensorDetails)}]</span>
                    {tensorDetails.description ? (
                      <span className="block ml-4 text-xs text-muted-foreground">{tensorDetails.description}</span>
                    ) : null}
                  </li>
                )
              })}
            </ul>
          ) : (
            <p>No inputs specified.</p>
          )}
        </div>

        <div>
          <p className="text-foreground font-medium">Outputs</p>
          {outputEntries.length ? (
            <ul className="ml-4 list-disc space-y-2">
              {outputEntries.map(([name, tensor]) => {
                const tensorDetails = tensor as Tensor
                return (
                  <li key={name}>
                    <span className="font-mono text-foreground">{name}</span>: {tensorDetails.dtype}
                    <span className="ml-2 text-muted-foreground">[{formatTensorShape(tensorDetails)}]</span>
                    {tensorDetails.description ? (
                      <span className="block ml-4 text-xs text-muted-foreground">{tensorDetails.description}</span>
                    ) : null}
                  </li>
                )
              })}
            </ul>
          ) : (
            <p>No outputs specified.</p>
          )}
        </div>

        {constraintEntries.length > 0 && (
          <div>
            <p className="text-foreground font-medium">Constraints</p>
            <ul className="ml-4 list-disc space-y-1">
              {constraintEntries.map((constraint, index) => (
                <li key={`${constraint}-${index}`} className="break-words">
                  {constraint}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  ) : null

  const solutionPanel = isSolution && solutionData ? (
    <Card className="p-6 flex h-full flex-col space-y-4 overflow-hidden">
      <h3 className="text-lg font-semibold">Solution Summary</h3>

      <div className="space-y-3 text-sm text-muted-foreground break-words">
        <div>
          <span className="text-foreground font-medium">Name:</span>{" "}
          <span className="font-mono text-foreground">{solutionData.name}</span>
        </div>
        <div>
          <span className="text-foreground font-medium">Definition:</span>{" "}
          {solutionData.definition ? (
            <Link
              href={`/kernels/${encodeURIComponent(solutionData.definition)}`}
              className="text-primary hover:underline inline-flex items-center gap-1"
            >
              <span>{solutionData.definition}</span>
            </Link>
          ) : (
            "-"
          )}
        </div>
        <div>
          <span className="text-foreground font-medium">Author:</span>{" "}
          <span>{solutionData.author}</span>
        </div>
        {spec && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-foreground font-medium">Language:</span>
            <Badge variant="outline" className="uppercase tracking-wide">
              {spec.language}
            </Badge>
          </div>
        )}
        {spec?.binding && (
          <div>
            <span className="text-foreground font-medium">C++ Binding:</span>{" "}
            <span>{spec.binding}</span>
          </div>
        )}
        {spec && (
          <div>
            <span className="text-foreground font-medium">Destination Passing Style:</span>{" "}
            <span>{spec.destination_passing_style === false ? "No" : spec.destination_passing_style === true ? "Yes" : "Yes (default)"}</span>
          </div>
        )}
        {solutionData.description && (
          <div>
            <span className="text-foreground font-medium">Description:</span>
            <p className="mt-1 whitespace-pre-wrap">{solutionData.description}</p>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-auto pr-2 text-sm text-muted-foreground space-y-4">
        {targetHardware.length > 0 && (
          <div>
            <p className="text-foreground font-medium">Target Hardware</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {targetHardware.map((target) => (
                <Badge key={target} variant="outline">
                  {target}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {spec?.entry_point && (
          <div>
            <p className="text-foreground font-medium">Entry Point</p>
            <p className="mt-1 font-mono text-foreground break-words">{spec.entry_point}</p>
          </div>
        )}

        {sourceCount > 0 && (
          <div>
            <p className="text-foreground font-medium">Source Files</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {Object.keys(sourceCode).map((path) => (
                <Button
                  key={path}
                  variant={activeSourceFile === path ? "default" : "outline"}
                  size="sm"
                  className="h-8 justify-start font-mono"
                  onClick={() => setActiveSourceFile(path)}
                >
                  {path}
                </Button>
              ))}
            </div>
          </div>
        )}

        {dependencies.length > 0 && (
          <div>
            <p className="text-foreground font-medium">Dependencies</p>
            <ul className="ml-4 list-disc space-y-1">
              {dependencies.map((dependency) => (
                <li key={dependency} className="break-words">
                  {dependency}
                </li>
              ))}
            </ul>
          </div>
        )}

        {buildCommands.length > 0 && (
          <div>
            <p className="text-foreground font-medium">Build Commands</p>
            <ul className="ml-4 list-disc space-y-1">
              {buildCommands.map((command, index) => (
                <li key={`${command}-${index}`} className="break-words font-mono text-xs text-foreground">
                  {command}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  ) : null

  const jsonPanel = (
    <Card className="p-4 flex flex-col h-full">
      <div>
        <h3 className="text-lg font-semibold flex items-center gap-2 mb-2">
          <FileText className="h-5 w-5" />
          JSON Configuration (Live Preview)
        </h3>

        <div className="flex flex-wrap gap-2 mb-4">
          {isDefinition ? (
            <>
              <Button size="sm" variant="outline" onClick={insertAxis}>
                <Plus className="h-3 w-3 mr-1" />
                Add Axis
              </Button>
              <Button size="sm" variant="outline" onClick={insertInput}>
                <Plus className="h-3 w-3 mr-1" />
                Add Input
              </Button>
              <Button size="sm" variant="outline" onClick={insertOutput}>
                <Plus className="h-3 w-3 mr-1" />
                Add Output
              </Button>
              <Button size="sm" variant="outline" onClick={insertConstraint}>
                <Plus className="h-3 w-3 mr-1" />
                Add Constraint
              </Button>
            </>
          ) : (
            <>
              <Button size="sm" variant="outline" onClick={insertDependency}>
                <Plus className="h-3 w-3 mr-1" />
                Add Dependency
              </Button>
              <Button size="sm" variant="outline" onClick={insertTargetHardware}>
                <Plus className="h-3 w-3 mr-1" />
                Add Target Hardware
              </Button>
            </>
          )}
        </div>
      </div>

      {jsonError && (
        <div className="text-sm text-destructive bg-destructive/10 p-2 rounded mb-4">
          JSON Error: {jsonError}
        </div>
      )}

      <div className="border rounded-lg overflow-hidden flex-1">
        <MonacoEditor
          height="100%"
          defaultLanguage="json"
          value={jsonText}
          onChange={(value) => setJsonText(value || "")}
          theme="vs-dark"
          options={{
            minimap: { enabled: false },
            fontSize: 14,
            lineNumbers: "on",
            wordWrap: "on",
            automaticLayout: true,
            formatOnPaste: true,
            formatOnType: true,
          }}
        />
      </div>
    </Card>
  )

  const metaPanel = definitionPanel ?? solutionPanel ?? jsonPanel

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Button variant="ghost" onClick={onBack} className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>
        <div className="flex gap-2">
          <Button variant="outline" onClick={copyToClipboard} className="gap-2">
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            {copied ? "Copied!" : "Copy JSON"}
          </Button>
          <Button onClick={exportToJson} className="gap-2">
            <Download className="h-4 w-4" />
            Export JSON
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2" style={{ height: "calc(100vh - 200px)" }}>
        {metaPanel}
        {codePanel}
      </div>
    </div>
  )
}

type TraceViewerProps = {
  data: Partial<Trace>
  onBack: () => void
}

function formatNumber(value: number | null | undefined, digits = 3) {
  if (value == null || Number.isNaN(value)) return "-"
  if (!Number.isFinite(value)) return String(value)
  const abs = Math.abs(value)
  if (abs >= 1 || abs === 0) return value.toFixed(digits)
  return value.toExponential(2)
}

function formatExtraValue(value: unknown): string {
  if (value == null) return "null"
  if (typeof value === "number") return formatNumber(value)
  if (typeof value === "boolean") return value ? "true" : "false"
  if (typeof value === "string") return value
  if (Array.isArray(value) || typeof value === "object") return JSON.stringify(value)
  return String(value)
}

function TraceViewer({ data, onBack }: TraceViewerProps) {
  const evaluation = data?.evaluation ?? null
  const performance = evaluation?.performance ?? null
  const correctness = evaluation?.correctness ?? null
  const environment = evaluation?.environment ?? null
  const axes = (data?.workload?.axes ?? {}) as Record<string, number>
  const inputs = (data?.workload?.inputs ?? {}) as Record<string, WorkloadInput>
  const libs = (environment?.libs ?? {}) as Record<string, string>
  const definitionName = data?.definition || ""
  const log = typeof evaluation?.log === "string" ? evaluation.log : null
  const showLogPanel = log !== null && log !== ""
  const correctnessExtra = correctness ? correctness.extra ?? null : null

  const status = evaluation?.status || "N/A"
  const statusTone = status === "PASSED" ? "text-emerald-600" : status.includes("INCORRECT") ? "text-amber-600" : status.includes("ERROR") ? "text-red-600" : "text-muted-foreground"

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Button variant="ghost" onClick={onBack} className="gap-2">
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-6 space-y-4">
          <h3 className="text-lg font-semibold">Trace Summary</h3>
          <div className="space-y-3 text-sm text-muted-foreground break-words">
            <div>
              <span className="text-foreground font-medium">Definition:</span>{" "}
              {definitionName ? (
                <Link href={`/kernels/${encodeURIComponent(definitionName)}`} className="text-primary hover:underline inline-flex items-center gap-1">
                  <span>{definitionName}</span>
                </Link>
              ) : (
                "-"
              )}
            </div>
            <div>
              <span className="text-foreground font-medium">Solution:</span> {data.solution ?? "Workload only"}
            </div>
            <div>
              <span className="text-foreground font-medium">Status:</span>{" "}
              <span className={`font-semibold ${statusTone}`}>{status}</span>
            </div>
            {evaluation?.timestamp && (
              <div>
                <span className="text-foreground font-medium">Timestamp:</span> {evaluation.timestamp}
              </div>
            )}
            {typeof log === "string" && (
              <div>
                <span className="text-foreground font-medium">Log:</span>
                {log === "" ? (
                  <span className="ml-1 font-mono text-muted-foreground">empty</span>
                ) : (
                  <span className="ml-1 text-muted-foreground">View log below.</span>
                )}
              </div>
            )}
            {performance && (
              <div className="pt-1">
                <p className="text-foreground font-medium">Performance</p>
                <ul className="ml-4 list-disc space-y-1">
                  <li>Latency: {formatNumber(performance.latency_ms)} ms</li>
                  <li>Reference latency: {formatNumber(performance.reference_latency_ms)} ms</li>
                  <li>Speedup factor: {formatNumber(performance.speedup_factor)}</li>
                </ul>
              </div>
            )}
            {correctness && (
              <div className="pt-1">
                <p className="text-foreground font-medium">Correctness</p>
                <ul className="ml-4 list-disc space-y-1">
                  <li>Max absolute error: {formatNumber(correctness.max_absolute_error)}</li>
                  <li>Max relative error: {formatNumber(correctness.max_relative_error)}</li>
                  {correctnessExtra && typeof correctnessExtra === "object" && Object.keys(correctnessExtra).length > 0 && (
                    <li>
                      Extra:
                      <ul className="ml-4 list-disc space-y-1">
                        {Object.entries(correctnessExtra).map(([key, value]) => (
                          <li key={key}>
                            <span className="font-mono text-foreground">{key}</span>:{" "}
                            <span className="break-all">{formatExtraValue(value)}</span>
                          </li>
                        ))}
                      </ul>
                    </li>
                  )}
                </ul>
              </div>
            )}
            {environment && (
              <div className="pt-1">
                <p className="text-foreground font-medium">Environment</p>
                <ul className="ml-4 list-disc space-y-1">
                  <li>Hardware: {environment.hardware || "-"}</li>
                  {Object.keys(libs).length > 0 && (
                    <li>
                      Libraries:
                      <ul className="ml-4 list-disc space-y-1">
                        {Object.entries(libs).map(([name, version]) => (
                          <li key={name}>
                            <span className="font-semibold text-foreground">{name}</span>: <span className="break-all">{version}</span>
                          </li>
                        ))}
                      </ul>
                    </li>
                  )}
                </ul>
              </div>
            )}
          </div>
        </Card>

        <Card className="p-6 space-y-4">
          <h3 className="text-lg font-semibold">Workload</h3>
          <div className="space-y-4 text-sm text-muted-foreground break-words">
            <div>
              <p className="text-foreground font-medium">Axes</p>
              {Object.keys(axes).length ? (
                <ul className="ml-4 list-disc space-y-1">
                  {Object.entries(axes).map(([name, value]) => (
                    <li key={name}>
                      <span className="font-mono text-foreground">{name}</span>: {value}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No axes information.</p>
              )}
            </div>
            <div>
              <p className="text-foreground font-medium">Inputs</p>
              {Object.keys(inputs).length ? (
                <ul className="ml-4 list-disc space-y-2">
                  {Object.entries(inputs).map(([name, input]) => (
                    <li key={name}>
                      <span className="font-mono text-foreground">{name}</span>: {input.type}
                      {input.type === "safetensors" && (
                        <span className="block ml-4 break-all">{input.path}::{input.tensor_key}</span>
                      )}
                      {input.type === "scalar" && (
                        <span className="block ml-4">value = {String(input.value)}</span>
                      )}
                      {input.type === "random" && input.seed != null && (
                        <span className="block ml-4">seed = {input.seed}</span>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No input descriptors.</p>
              )}
            </div>
          </div>
        </Card>
      </div>

      {showLogPanel && (
        <Card className="p-4 space-y-3">
          <h3 className="text-lg font-semibold">Log</h3>
          <div className="border rounded-lg overflow-hidden">
            <MonacoEditor
              height="320px"
              defaultLanguage="plaintext"
              value={log}
              theme="vs-dark"
              options={{
                readOnly: true,
                minimap: { enabled: false },
                fontSize: 13,
                lineNumbers: "on",
                wordWrap: "on",
                automaticLayout: true,
              }}
            />
          </div>
        </Card>
      )}
    </div>
  )
}
