import Link from "next/link"
import type { ReactNode } from "react"

import { cn } from "@flashinfer-bench/utils"

import { Logo } from "../brand/Logo"

export interface SiteHeaderNavItem {
  href: string
  label: string
  external?: boolean
}

export interface SiteHeaderProps {
  className?: string
  navItems?: SiteHeaderNavItem[]
  searchSlot?: ReactNode
  rightSlot?: ReactNode
  logoHref?: string
  logoHeight?: number
}

export function SiteHeader({
  className,
  navItems = [],
  searchSlot,
  rightSlot,
  logoHref = "/",
  logoHeight = 40,
}: SiteHeaderProps) {
  return (
    <header
      className={cn(
        "sticky top-0 z-50 w-full border-b border-border/60 bg-background/95 backdrop-blur",
        "supports-[backdrop-filter]:bg-background/60",
        className,
      )}
    >
      <div className="mx-auto flex h-16 w-full max-w-[1400px] items-center px-4 md:px-6">
        <Link
          href={logoHref}
          className="flex shrink-0 items-center"
          aria-label="FlashInfer Bench"
        >
          <Logo height={logoHeight} />
        </Link>

        <div className="flex flex-1 items-center justify-end gap-4 md:gap-6">
          {searchSlot ? (
            <div className="hidden items-center md:flex md:min-w-[240px] md:max-w-[320px]">
              {searchSlot}
            </div>
          ) : null}

          {navItems.length > 0 ? (
            <nav className="flex items-center gap-4 md:gap-6">
              {navItems.map(({ href, label, external }) => {
                const classNames = "text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
                if (external) {
                  return (
                    <a
                      key={href}
                      href={href}
                      className={classNames}
                    >
                      {label}
                    </a>
                  )
                }
                return (
                  <Link key={href} href={href} className={classNames}>
                    {label}
                  </Link>
                )
              })}
            </nav>
          ) : null}

          {rightSlot ? (
            <div className="flex items-center gap-3">{rightSlot}</div>
          ) : null}
        </div>
      </div>
    </header>
  )
}
