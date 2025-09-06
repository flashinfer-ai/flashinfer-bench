"use client"

import { useState, useEffect, useCallback } from "react"
import { Editor as MonacoEditor } from "@monaco-editor/react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ArrowLeft, Copy, Download, Check, Plus, FileText, Code } from "lucide-react"
import { toast } from "@/components/ui/toaster"

interface EditorProps {
  data: any
  onBack: () => void
}

export function Editor({ data, onBack }: EditorProps) {
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

  const insertSourceFile = () => {
    const newFileName = `new_file_${Object.keys(sourceCode).length + 1}.py`
    setSourceCode(prev => ({
      ...prev,
      [newFileName]: "# New source file\n"
    }))
    setActiveSourceFile(newFileName)
  }

  const removeSourceFile = (path: string) => {
    const newSourceCode = { ...sourceCode }
    delete newSourceCode[path]
    setSourceCode(newSourceCode)

    // Switch to another file if the active one was removed
    if (path === activeSourceFile) {
      const remaining = Object.keys(newSourceCode)
      setActiveSourceFile(remaining[0] || "")
    }
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

      <div className="grid grid-cols-2 gap-4" style={{ height: "calc(100vh - 200px)" }}>
        {/* Code Editor on Left */}
        <Card className="p-4 flex flex-col h-full">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Code className="h-5 w-5" />
              {isDefinition ? "Reference Implementation" : "Source Code"}
            </h3>
            {isSolution && (
              <Button size="sm" onClick={insertSourceFile} className="gap-2">
                <Plus className="h-4 w-4" />
                Add File
              </Button>
            )}
          </div>

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
            ) : isSolution && Object.keys(sourceCode).length > 0 ? (
              <div className="flex flex-col h-full space-y-2">
                <div className="flex gap-2 flex-wrap">
                  {Object.keys(sourceCode).map((path) => (
                    <div key={path} className="flex items-center gap-1">
                      <Button
                        size="sm"
                        variant={activeSourceFile === path ? "default" : "outline"}
                        onClick={() => setActiveSourceFile(path)}
                      >
                        {path}
                      </Button>
                      {Object.keys(sourceCode).length > 1 && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => removeSourceFile(path)}
                          className="h-8 w-8 p-0"
                        >
                          ×
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
                {activeSourceFile && (
                  <div className="border rounded-lg overflow-hidden flex-1">
                    <MonacoEditor
                      height="100%"
                      language={activeSourceFile.endsWith('.py') ? 'python' : 'plaintext'}
                      value={sourceCode[activeSourceFile] || ""}
                      onChange={(value) => setSourceCode(prev => ({
                        ...prev,
                        [activeSourceFile]: value || ""
                      }))}
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
                )}
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-muted-foreground">
                No code content found.
              </div>
            )}
          </div>
        </Card>

        {/* JSON Configuration on Right */}
        <Card className="p-4 flex flex-col h-full">
          <div>
            <h3 className="text-lg font-semibold flex items-center gap-2 mb-2">
              <FileText className="h-5 w-5" />
              JSON Configuration (Live Preview)
            </h3>

            {/* Quick Action Buttons */}
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
      </div>
    </div>
  )
}
