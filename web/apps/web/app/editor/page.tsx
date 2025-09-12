"use client"

import { useState, useEffect, Suspense } from "react"
import { useSearchParams } from "next/navigation"
import stripJsonComments from "strip-json-comments"
import { Editor } from "@/components/editor"
import { Button, Textarea, Card } from "@flashinfer-bench/ui"
import { AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@flashinfer-bench/ui"

function EditorContent() {
  const [jsonInput, setJsonInput] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [parsedData, setParsedData] = useState<any>(null)
  const searchParams = useSearchParams()

  useEffect(() => {
    // Check for solution ID in URL params
    const solutionId = searchParams.get('solution')
    if (solutionId) {
      // Try to get solution data from sessionStorage
      const storedSolution = sessionStorage.getItem(`solution-${solutionId}`)
      if (storedSolution) {
        try {
          const data = JSON.parse(storedSolution)
          setParsedData(data)
          setError(null)
          // Clean up sessionStorage
          sessionStorage.removeItem(`solution-${solutionId}`)
        } catch (e) {
          console.error('Failed to parse stored solution:', e)
        }
      }
    }
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

      window.history.pushState({ view: 'editor' }, '', window.location.href)
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

  return (
    <div className={parsedData ? "py-4 px-6" : "container mx-auto py-8 px-4 max-w-7xl"}>
      {!parsedData && (
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Definition & Solution Editor</h1>
          <p className="text-muted-foreground">
            View and edit definitions and solutions. Paste a complete definition or solution JSON to get started.
          </p>
        </div>
      )}

      {!parsedData ? (
        <Card className="p-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="json-input" className="block text-sm font-medium mb-2">
                Paste Definition or Solution JSON
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
          <Editor
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

export default function EditorPage() {
  return (
    <Suspense fallback={<div className="container mx-auto py-8 px-4 max-w-7xl">Loading...</div>}>
      <EditorContent />
    </Suspense>
  )
}
