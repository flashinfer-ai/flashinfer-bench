"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Maximize2, Minimize2, Copy, Check } from "lucide-react"
import { Definition } from "@/lib/schemas"
import dynamic from "next/dynamic"

const MonacoEditor = dynamic(
  () => import("@monaco-editor/react"),
  {
    ssr: false,
    loading: () => <div className="h-[300px] flex items-center justify-center">Loading editor...</div>
  }
)

export function DefinitionDetails({ definition }: { definition: Definition }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [showExpandButton, setShowExpandButton] = useState(false)
  const [copied, setCopied] = useState(false)
  const editorRef = useRef<any>(null)

  const handleEditorDidMount = (editor: any) => {
    editorRef.current = editor

    // Check if content needs scrolling
    const checkScrollNeeded = () => {
      const contentHeight = editor.getContentHeight()
      const containerHeight = 300 // Default height
      setShowExpandButton(contentHeight > containerHeight)
    }

    checkScrollNeeded()
    editor.onDidChangeModelContent(checkScrollNeeded)
  }

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(definition.reference)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error("Failed to copy:", err)
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={copyToClipboard}
            className="h-8 px-2"
          >
            {copied ? (
              <>
                <Check className="h-4 w-4 mr-1" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-4 w-4 mr-1" />
                Copy
              </>
            )}
          </Button>
          {showExpandButton && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="h-8 px-2"
            >
              {isExpanded ? (
                <>
                  <Minimize2 className="h-4 w-4 mr-1" />
                  Collapse
                </>
              ) : (
                <>
                  <Maximize2 className="h-4 w-4 mr-1" />
                  Expand
                </>
              )}
            </Button>
          )}
        </div>
      </div>
      <div className="border rounded-md overflow-hidden">
        <MonacoEditor
          height={isExpanded ? "1080px" : "600px"}
          language="python"
          theme="vs-dark"
          value={definition.reference}
          onMount={handleEditorDidMount}
          options={{
            readOnly: true,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            fontSize: 14,
            lineNumbers: "on",
            folding: false,
            lineDecorationsWidth: 10,
            lineNumbersMinChars: 3,
            padding: { top: 16, bottom: 16 },
            renderLineHighlight: "none",
            overviewRulerLanes: 0,
            hideCursorInOverviewRuler: true,
            scrollbar: {
              vertical: "visible",
              horizontal: "visible",
              verticalScrollbarSize: 10,
              horizontalScrollbarSize: 10
            }
          }}
        />
      </div>
    </div>
  )
}
