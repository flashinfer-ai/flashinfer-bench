import Link from "next/link"
import { Github } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t bg-background">
      <div className="container flex flex-col items-center justify-between gap-4 py-10 md:h-24 md:flex-row md:py-0">
        <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
          <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
            Built by the FlashInfer community.
          </p>
        </div>
        <div className="flex items-center space-x-1">
          <Link
            href="https://github.com/flashinfer-ai"
            target="_blank"
            rel="noreferrer"
          >
            <div className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background hover:bg-accent hover:text-accent-foreground h-9 py-2 px-3">
              <Github className="h-4 w-4" />
              <span className="sr-only">GitHub</span>
            </div>
          </Link>
        </div>
      </div>
    </footer>
  )
}