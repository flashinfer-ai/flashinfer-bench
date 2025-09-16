import { Card, CardContent } from "@flashinfer-bench/ui"
import { Definition } from "@/lib/schemas"

export function ConstraintsSection({ definition }: { definition: Definition }) {
  if (!definition.constraints || definition.constraints.length === 0) return null
  return (
    <section id="constraints">
      <h2 className="text-2xl font-semibold mb-4">Constraints</h2>
      <Card>
        <CardContent className="pt-6">
          <ul className="space-y-2">
            {definition.constraints.map((constraint, idx) => (
              <li key={idx} className="text-sm font-mono text-muted-foreground">
                • {constraint}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </section>
  )
}
