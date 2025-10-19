import { cn } from "@flashinfer-bench/utils"

export type FastPLabelProps = {
  className?: string
  value?: string | number
}

export function FastPLabel({ className, value }: FastPLabelProps) {
  const subscript = value !== undefined ? String(value) : "p"
  return (
    <span
      className={cn("inline-flex items-baseline gap-0.5", className)}
      aria-label={`fast sub ${subscript}`}
    >
      <span>fast</span>
      <sub className="text-[0.65em] leading-none">{subscript}</sub>
    </span>
  )
}
