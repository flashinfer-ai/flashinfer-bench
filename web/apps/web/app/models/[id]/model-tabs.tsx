"use client"

import dynamic from "next/dynamic"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@flashinfer-bench/ui"
import { ModelVisualization } from "./model-visualization"
import { ModelBarChart } from "./model-bar-chart"
import { Model } from "@/lib/schemas"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@flashinfer-bench/ui"

// Dynamic import for client-side only rendering
const ModelFlowWrapper = dynamic(
  () => import("./model-flow-visualization").then(mod => mod.ModelFlowWrapper),
  {
    ssr: false,
    loading: () => <div className="h-[800px] flex items-center justify-center">Loading visualization...</div>
  }
)

export function ModelTabs({ model }: { model: Model }) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-semibold mb-4">Architecture Overview</h2>
        <Tabs defaultValue="flow" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="flow">Flow View</TabsTrigger>
            <TabsTrigger value="compute">Module Compute Contribution</TabsTrigger>
          </TabsList>

          <TabsContent value="flow" className="mt-6">
            <ModelFlowWrapper model={model} />
          </TabsContent>

          <TabsContent value="compute" className="mt-6">
            <ModelBarChart model={model} />
          </TabsContent>
        </Tabs>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Architecture Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-2xl font-bold">
                {Object.keys(model.modules).length}
              </p>
              <p className="text-sm text-muted-foreground">Total Modules</p>
            </div>
            <div>
              <p className="text-2xl font-bold">
                {Object.values(model.modules).filter(m => m.type === "block").length}
              </p>
              <p className="text-sm text-muted-foreground">Blocks</p>
            </div>
            <div>
              <p className="text-2xl font-bold">
                {Object.values(model.modules).filter(m => m.type === "layer").length}
              </p>
              <p className="text-sm text-muted-foreground">Kernels</p>
            </div>
            <div>
              <p className="text-2xl font-bold">
                {Object.values(model.modules).filter(m => m.type === "layer" && m.definition).length}
              </p>
              <p className="text-sm text-muted-foreground">Traced Kernels</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div>
        <h2 className="text-2xl font-semibold mb-4">Hierarchy View</h2>
        <ModelVisualization model={model} />
      </div>
    </div>
  )
}
