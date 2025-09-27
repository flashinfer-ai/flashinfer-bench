"use client"

import dynamic from "next/dynamic"
import { Model } from "@/lib/schemas"
import { Card, CardContent, CardHeader, CardTitle } from "@flashinfer-bench/ui"

// Dynamic import for client-side only rendering
const ModelArchWrapper = dynamic(
  () => import("./arch-overview").then(mod => mod.ModelArchWrapper),
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
        <ModelArchWrapper model={model} />
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
                {Object.values(model.modules).filter(m => m.type === "layer" && (m.definitions?.length ?? 0) > 0).length}
              </p>
              <p className="text-sm text-muted-foreground">Traced Kernels</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
