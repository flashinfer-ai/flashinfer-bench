"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@flashinfer-bench/ui"
import { Copy, Check, ArrowLeft } from "lucide-react"
import { Definition } from "@/lib/schemas"

export function DefinitionHeader({
  definition,
  solutionsCount,
}: {
  definition: Definition
  solutionsCount: number
}) {
  const [copiedItem, setCopiedItem] = useState<string | null>(null)

  const copyToClipboard = async (text: string, type: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedItem(type)
      setTimeout(() => setCopiedItem(null), 2000)
    } catch (err) {
      console.error("Failed to copy:", err)
    }
  }

  const copyJSON = () => {
    const orderedDefinition: any = {}
    Object.keys(definition).forEach((key) => {
      orderedDefinition[key] = (definition as any)[key]
    })
    copyToClipboard(JSON.stringify(orderedDefinition, null, 2), "json")
  }

  return (
    <div className="sticky top-14 z-40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
      <div className="container py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-4 w-4" />
            </Link>
            <h1 className="text-xl font-mono font-bold">{definition.name}</h1>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Button variant="ghost" size="sm" onClick={copyJSON}>
              {copiedItem === "json" ? (
                <>
                  <Check className="h-3 w-3 mr-1" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3 mr-1" />
                  Copy JSON
                </>
              )}
            </Button>
            <span className="text-muted-foreground">·</span>
            <a href="#solutions" className="hover:underline">
              Solutions ({solutionsCount})
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
