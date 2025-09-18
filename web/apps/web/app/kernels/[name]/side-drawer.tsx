"use client"

import type { ReactNode } from "react"

export type SideDrawerProps = {
  title: string
  open: boolean
  onClose: () => void
  children: ReactNode
  side?: "left" | "right"
}

export function SideDrawer({ title, open, onClose, children, side = "left" }: SideDrawerProps) {
  return (
    <div className={`fixed inset-0 z-40 ${open ? "" : "pointer-events-none"}`}>
      <div className={`absolute inset-0 bg-black/30 transition-opacity ${open ? "opacity-100" : "opacity-0"}`} onClick={onClose} />
      <div
        className={`absolute top-0 ${side === "left" ? "left-0" : "right-0"} h-full w-[360px] bg-background border ${open ? "translate-x-0" : side === "left" ? "-translate-x-full" : "translate-x-full"} transition-transform`}
      >
        <div className="p-4 border-b font-medium">{title}</div>
        <div className="p-4 space-y-4 overflow-auto h-[calc(100%-48px)]">{children}</div>
      </div>
    </div>
  )
}
