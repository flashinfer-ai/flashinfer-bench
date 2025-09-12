import Link from "next/link"
import { Logo } from "@flashinfer-bench/ui/brand/Logo"

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur-sm supports-backdrop-filter:bg-background/60">
      <div className="container flex h-14 items-center">
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <span className="hidden font-bold sm:inline-block">
              <Logo />
            </span>
          </Link>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            <Link
              href="/models"
              className="transition-colors hover:text-foreground/80 text-foreground/60"
            >
              Models
            </Link>
            <Link
              href="/editor"
              className="transition-colors hover:text-foreground/80 text-foreground/60"
            >
              Editor
            </Link>
          </nav>
        </div>
      </div>
    </header>
  )
}
