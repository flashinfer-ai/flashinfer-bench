"use client"

import { useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ChevronRight, Package, Layers } from "lucide-react"
import { Model, ModelHierarchy } from "@/lib/schemas"
import { getRootModules, getChildren } from "@/lib/model-utils"

export function ModelVisualization({ model }: { model: Model }) {
  const [selectedPath, setSelectedPath] = useState<string[]>([])
  
  // Build hierarchy from flat module structure
  function buildHierarchy(moduleName: string): ModelHierarchy {
    const moduleData = model.modules[moduleName]
    if (!moduleData) return { name: moduleName, type: "layer" }
    
    const childNames = getChildren(model, moduleName)
    const children = childNames.map(childName => buildHierarchy(childName))
    
    return {
      name: moduleName,
      type: moduleData.type,
      definition: moduleData.definition,
      children: children.length > 0 ? children : undefined
    }
  }
  
  // Get root modules (no parent)
  const rootModuleNames = getRootModules(model)
  const rootModules = rootModuleNames.map(name => buildHierarchy(name))
  
  // Get current view based on selected path
  function getCurrentModules(): ModelHierarchy[] {
    if (selectedPath.length === 0) return rootModules
    
    let current = rootModules
    for (const pathItem of selectedPath) {
      const found = current.find(m => m.name === pathItem)
      if (found?.children) {
        current = found.children
      }
    }
    return current
  }
  
  const currentModules = getCurrentModules()
  
  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center space-x-2 text-sm">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setSelectedPath([])}
          className="px-2"
        >
          {model.name}
        </Button>
        {selectedPath.map((path, idx) => (
          <div key={idx} className="flex items-center">
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelectedPath(selectedPath.slice(0, idx + 1))}
              className="px-2"
            >
              {path}
            </Button>
          </div>
        ))}
      </div>
      
      {/* Current level visualization */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {currentModules.map((hierarchyNode) => {
          const moduleData = model.modules[hierarchyNode.name]
          const hasChildren = hierarchyNode.children && hierarchyNode.children.length > 0
          
          return (
            <Card 
              key={hierarchyNode.name}
              className={`transition-all ${hasChildren ? 'hover:shadow-lg cursor-pointer' : ''}`}
              onClick={() => {
                if (hasChildren) {
                  setSelectedPath([...selectedPath, hierarchyNode.name])
                }
              }}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {hierarchyNode.type === "block" ? (
                      <Package className="h-5 w-5 text-muted-foreground" />
                    ) : (
                      <Layers className="h-5 w-5 text-muted-foreground" />
                    )}
                    <CardTitle className="text-lg">{hierarchyNode.name}</CardTitle>
                  </div>
                  {hasChildren && (
                    <ChevronRight className="h-5 w-5 text-muted-foreground" />
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={hierarchyNode.type === "block" ? "default" : "secondary"}>
                    {hierarchyNode.type}
                  </Badge>
                  {moduleData?.count && moduleData.count > 1 && (
                    <Badge variant="outline">×{moduleData.count}</Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {hierarchyNode.definition && (
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">Kernel Definition</p>
                    <Link 
                      href={`/kernels/${hierarchyNode.definition}`}
                      className="text-sm font-mono text-primary hover:underline"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {hierarchyNode.definition}
                    </Link>
                  </div>
                )}
                {hierarchyNode.children && (
                  <p className="text-sm text-muted-foreground mt-2">
                    {hierarchyNode.children.length} sub-modules
                  </p>
                )}
                {!hierarchyNode.definition && !hierarchyNode.children && (
                  <p className="text-sm text-muted-foreground">
                    Base layer implementation
                  </p>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}