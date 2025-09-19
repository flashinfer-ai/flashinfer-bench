"use client"

import { useState, useEffect, Suspense } from "react"
import { useSearchParams } from "next/navigation"
import stripJsonComments from "strip-json-comments"
import { Viewer } from "@/components/viewer"
import { Button, Textarea, Card } from "@flashinfer-bench/ui"
import { AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@flashinfer-bench/ui"

function ViewerContent() {
  const [jsonInput, setJsonInput] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [parsedData, setParsedData] = useState<any>(null)
  const searchParams = useSearchParams()

  useEffect(() => {
    const loadFromStorage = (prefix: string, id: string | null) => {
      if (!id) return false
      const key = `${prefix}-${id}`
      const stored = sessionStorage.getItem(key)
      if (!stored) {
        setError("Stored viewer data not found. Please reopen from the results view.")
        return false
      }
      try {
        const data = JSON.parse(stored)
        setParsedData(data)
        setError(null)
        sessionStorage.removeItem(key)
        return true
      } catch (e) {
        console.error(`Failed to parse stored ${prefix}:`, e)
        setError("Unable to read stored viewer data. Please reopen from the results view.")
        sessionStorage.removeItem(key)
        return false
      }
    }

    const solutionId = searchParams.get("solution")
    if (loadFromStorage("solution", solutionId)) return

    const traceId = searchParams.get("trace")
    loadFromStorage("trace", traceId)
  }, [searchParams])

  useEffect(() => {
    const handlePopState = () => {
      setParsedData(null)
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  const handleParse = () => {
    try {
      const cleanedJson = stripJsonComments(jsonInput, { trailingCommas: true })
      const data = JSON.parse(cleanedJson)
      setParsedData(data)
      setError(null)

      window.history.pushState({ view: 'viewer' }, '', window.location.href)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Invalid JSON format. Please check your input.")
      setParsedData(null)
    }
  }

  const handleClear = () => {
    setJsonInput("")
    setParsedData(null)
    setError(null)
  }

  const isTraceData = parsedData && typeof parsedData === "object" && "workload" in parsedData

  const containerClass = parsedData
    ? isTraceData
      ? "container mx-auto py-8 px-4"
      : "py-4 px-6"
    : "container mx-auto py-8 px-4 max-w-7xl"

  return (
    <div className={containerClass}>
      {!parsedData && (
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Definition, Solution & Trace Viewer</h1>
          <p className="text-muted-foreground">
            Inspect definitions, solutions, and traces in a friendly layout. Paste a complete JSON payload or open items directly from the results page.
          </p>
        </div>
      )}

      {!parsedData ? (
        <Card className="p-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="json-input" className="block text-sm font-medium mb-2">
                Paste Definition, Solution, or Trace JSON
              </label>
              <Textarea
                id="json-input"
                placeholder="Paste your JSON here..."
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                className="font-mono text-sm min-h-[400px]"
              />
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="flex gap-2">
              <Button onClick={handleParse} disabled={!jsonInput.trim()}>
                Parse JSON
              </Button>
              <Button variant="outline" onClick={handleClear}>
                Clear
              </Button>
            </div>
          </div>
        </Card>
      ) : (
        <div className="w-full -mx-4 px-4">
          <Viewer
            data={parsedData}
            onBack={() => {
              setParsedData(null)
            }}
          />
        </div>
      )}
    </div>
  )
}

export default function ViewerPage() {
  return (
    <Suspense fallback={<div className="container mx-auto py-8 px-4 max-w-7xl">Loading...</div>}>
      <ViewerContent />
    </Suspense>
  )
}
