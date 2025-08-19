"use client"

import React from "react"
import Link from "next/link"
import { Cpu, Copy, Check, Filter } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ProgressCircle } from "@/components/ui/progress-circle"
import { Button } from "@/components/ui/button"
import { Model } from "@/lib/schemas"

interface ModelCardProps {
  model: Model
  href: string
}

export function ModelCard({ model, href }: ModelCardProps) {
  const [copied, setCopied] = React.useState(false)
  
  // Count only layers (kernels)
  const totalKernels = Object.values(model.modules).filter(m => m.type === "layer").length
  const tracedKernels = Object.values(model.modules).filter(m => m.type === "layer" && m.definition).length
  
  const handleCopy = async (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    await navigator.clipboard.writeText(model.id)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  
  const handleFilter = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    window.location.href = `/?kernel_search=${encodeURIComponent(model.id)}`
  }
  
  return (
    <Link href={href}>
      <Card className="hover:shadow-lg hover:border-primary transition-all cursor-pointer h-full">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 flex-1 min-w-0">
              <Cpu className="h-5 w-5 text-muted-foreground flex-shrink-0" />
              <div className="group flex items-center gap-2 min-w-0">
                <CardTitle className="text-lg truncate">{model.name}</CardTitle>
                <button
                  onClick={handleCopy}
                  className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                  aria-label="Copy model ID"
                >
                  {copied ? (
                    <Check className="h-3 w-3 text-green-600" />
                  ) : (
                    <Copy className="h-3 w-3 text-muted-foreground hover:text-foreground" />
                  )}
                </button>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 flex-shrink-0"
              onClick={handleFilter}
            >
              <Filter className="h-3.5 w-3.5" />
            </Button>
          </div>
          <CardDescription>{model.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              <span>{totalKernels} kernels</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">
                {tracedKernels}/{totalKernels} traced
              </span>
              <ProgressCircle 
                value={tracedKernels} 
                max={totalKernels}
                size={24}
                strokeWidth={2.5}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}