"use client"

import { useCallback, useState, useEffect, useMemo } from "react"
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Handle,
  Position,
  NodeProps,
  ReactFlowProvider,
} from "reactflow"
import "reactflow/dist/style.css"
import Link from "next/link"
import { Badge, Card, CardContent } from "@flashinfer-bench/ui"
import { cn } from "@flashinfer-bench/utils"
import { Package, Layers, ChevronDown, ChevronRight } from "lucide-react"
import { Model, Module } from "@/lib/schemas"
import { getChildren } from "@/lib/model-utils"

// Custom node component
type ModelNodeData = {
  label: string
  moduleType: Module["type"]
  definitions?: string[]
  count: number
  hasParent: boolean
  children: string[]
  stats?: { sublayers: number; kernels: number }
  highlighted?: boolean
}

function ModelNode({ data, selected }: NodeProps<ModelNodeData>) {
  const [expanded, setExpanded] = useState(false)
  const hasChildren = data.children && data.children.length > 0
  const isBlock = data.moduleType === "block"
  const isTraced = (data.definitions?.length ?? 0) > 0

  const cardClasses = cn(
    "w-[250px] min-w-[250px] max-w-[250px] border transition-colors",
    isTraced ? "border-primary/50 bg-primary/5" : "border-border",
    (selected || data.highlighted) && "ring-2 ring-primary"
  )

  return (
    <Card className={cardClasses}>
      <CardContent className="p-3">
        <div className="space-y-2">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {isBlock ? (
                <Package className="h-4 w-4 text-muted-foreground" />
              ) : (
                <Layers className="h-4 w-4 text-muted-foreground" />
              )}
              <span className="font-medium text-sm">{data.label}</span>
            </div>
            {hasChildren && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="p-0.5 hover:bg-muted rounded"
              >
                {expanded ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </button>
            )}
          </div>

          {/* Badges */}
          <div className="flex gap-1 flex-wrap">
            <Badge variant={isBlock ? "default" : "secondary"} className="text-xs">
              {data.moduleType}
            </Badge>
            {data.count > 1 && (
              <Badge variant="outline" className="text-xs">×{data.count}</Badge>
            )}
          </div>

          {/* Kernel definitions */}
          {data.definitions && data.definitions.length > 0 && (
            <div className="pt-1 space-y-1">
              {data.definitions.map((definition) => (
                <div key={definition} className="text-xs leading-tight">
                  <Link
                    href={`/kernels/${definition}`}
                    className="text-primary hover:underline inline-flex max-w-full items-center gap-1"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <span className="truncate" title={definition}>
                      {definition.length > 34 ? `${definition.slice(0, 34)}…` : definition}
                    </span>
                  </Link>
                </div>
              ))}
            </div>
          )}

          {/* Stats if expanded */}
          {expanded && data.stats && (
            <div className="text-xs text-muted-foreground border-t pt-2 space-y-1">
              <div>Sublayers: {data.stats.sublayers}</div>
              <div>Kernels: {data.stats.kernels}</div>
            </div>
          )}
        </div>

        {/* Connection handles */}
        {data.hasParent && (
          <Handle
            type="target"
            position={Position.Top}
            id="target"
            className="w-2 h-2 border-none"
            style={{ background: isTraced ? "#2563eb" : "#9ca3af" }}
          />
        )}
        {hasChildren && (
          <Handle
            type="source"
            position={Position.Bottom}
            id="source"
            className="w-2 h-2 border-none"
            style={{ background: isTraced ? "#2563eb" : "#9ca3af" }}
          />
        )}
      </CardContent>
    </Card>
  )
}

// Moved inside component and memoized to avoid React Flow warnings

interface ModelFlowProps {
  model: Model
}

export function ModelArchOverview({ model }: ModelFlowProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState<ModelNodeData>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selectedModule, setSelectedModule] = useState<string | null>(null)

  // Memoize nodeTypes to avoid React Flow warnings
  const nodeTypes = useMemo(() => ({
    modelNode: ModelNode,
  }), [])

  // Build the graph structure
  useEffect(() => {
    const buildGraph = () => {
      const newNodes: Node<ModelNodeData>[] = []
      const newEdges: Edge[] = []
      const nodePositions = new Map<string, { x: number; y: number }>()

      // Helper to calculate subtree stats
      const getStats = (moduleName: string): { sublayers: number; kernels: number } => {
        const moduleData = model.modules[moduleName]
        if (!moduleData) return { sublayers: 0, kernels: 0 }

        let sublayers = 0
        let kernels = moduleData.definitions?.length ?? 0

        // Count children
        const children = getChildren(model, moduleName)
        for (const childName of children) {
          sublayers++
          const childStats = getStats(childName)
          sublayers += childStats.sublayers
          kernels += childStats.kernels
        }

        return { sublayers, kernels }
      }

      // Get hierarchy levels
      const getLevel = (moduleName: string): number => {
        const moduleData = model.modules[moduleName]
        if (!moduleData || !moduleData.parent) return 0
        return 1 + getLevel(moduleData.parent)
      }

      // Group modules by level
      const levels = new Map<number, string[]>()
      Object.entries(model.modules).forEach(([name, moduleData]) => {
        const level = getLevel(name)
        if (!levels.has(level)) levels.set(level, [])
        levels.get(level)!.push(name)
      })

      // Position nodes
      const levelHeight = 150
      const nodeWidth = 250
      const nodeSpacing = 50

      levels.forEach((moduleNames, level) => {
        const levelWidth = moduleNames.length * (nodeWidth + nodeSpacing)
        const startX = -levelWidth / 2 + nodeWidth / 2

        moduleNames.forEach((name, index) => {
          const x = startX + index * (nodeWidth + nodeSpacing)
          const y = level * levelHeight
          nodePositions.set(name, { x, y })
        })
      })

      // Create nodes first
      Object.entries(model.modules).forEach(([name, moduleData]) => {
        const pos = nodePositions.get(name) || { x: 0, y: 0 }
        const stats = getStats(name)
        const children = getChildren(model, name)

        newNodes.push({
          id: name,
          type: "modelNode",
          position: pos,
          data: {
            label: name,
            moduleType: moduleData.type,
            definitions: moduleData.definitions ?? [],
            count: moduleData.count || 1,
            hasParent: !!moduleData.parent,
            children,
            stats,
            highlighted: name === selectedModule,
          },
        })
      })

      // Create edges after all nodes are created
      Object.entries(model.modules).forEach(([name, moduleData]) => {
        if (moduleData.parent) {
          // Check if both source and target nodes exist
          const sourceExists = newNodes.some(n => n.id === moduleData.parent)
          const targetExists = newNodes.some(n => n.id === name)

          if (sourceExists && targetExists) {
            newEdges.push({
              id: `${moduleData.parent}-${name}`,
              source: moduleData.parent,
              sourceHandle: "source",
              target: name,
              targetHandle: "target",
              type: "smoothstep",
              style: {
                strokeWidth: 1.5,
                stroke: "#CBD5F5",
              },
            })
          }
        }
      })

      setNodes(newNodes)
      setEdges(newEdges)
    }

    buildGraph()
  }, [model, selectedModule, setNodes, setEdges])

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node<ModelNodeData>) => {
    setSelectedModule(node.id)
  }, [])

  return (
    <div className="h-[800px] w-full border rounded-lg overflow-hidden relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-right"
      >
        <Background variant={"dots" as any} gap={12} size={1} />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            if ((node.data?.definitions?.length ?? 0) > 0) return "#2563eb"
            return "#CBD5F5"
          }}
          nodeStrokeWidth={3}
          pannable
          zoomable
        />
      </ReactFlow>
    </div>
  )
}

export function ModelArchWrapper({ model }: { model: Model }) {
  return (
    <ReactFlowProvider>
      <ModelArchOverview model={model} />
    </ReactFlowProvider>
  )
}
