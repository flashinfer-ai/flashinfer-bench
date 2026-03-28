import { Badge } from "@flashinfer-bench/ui"

const TAG_STYLES: Record<string, { variant: "default" | "secondary" | "destructive" | "outline"; className: string }> = {
  stage:  { variant: "outline",   className: "border-blue-400 text-blue-700 dark:text-blue-300" },
  model:  { variant: "secondary", className: "" },
  tp:     { variant: "outline",   className: "border-violet-400 text-violet-700 dark:text-violet-300" },
  ep:     { variant: "outline",   className: "border-violet-400 text-violet-700 dark:text-violet-300" },
  fi_api: { variant: "outline",   className: "font-mono text-xs" },
}

export function TagsSection({ tags }: { tags: string[] }) {
  if (!tags || tags.length === 0) return null
  return (
    <div className="flex flex-wrap gap-2">
      {tags.map((tag) => {
        const colonIdx = tag.indexOf(":")
        const prefix = colonIdx !== -1 ? tag.slice(0, colonIdx) : ""
        const value  = colonIdx !== -1 ? tag.slice(colonIdx + 1) : tag

        let variant: "default" | "secondary" | "destructive" | "outline" = "outline"
        let className = ""

        if (prefix === "status") {
          variant = value === "draft" ? "destructive" : "default"
        } else if (prefix in TAG_STYLES) {
          variant   = TAG_STYLES[prefix].variant
          className = TAG_STYLES[prefix].className
        }

        return (
          <Badge key={tag} variant={variant} className={className}>
            {prefix ? (
              <><span className="opacity-50">{prefix}:</span>{value}</>
            ) : tag}
          </Badge>
        )
      })}
    </div>
  )
}
