import Link from "next/link"

const GITHUB_URL = "https://github.com/flashinfer-ai"

export function SiteFooter() {
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
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-9 items-center justify-center rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ring-offset-background"
          >
            <span className="sr-only">GitHub</span>
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              className="h-4 w-4"
              fill="currentColor"
            >
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.207 11.387.6.113.82-.262.82-.582 0-.288-.012-1.244-.018-2.256-3.338.726-4.042-1.61-4.042-1.61-.546-1.386-1.332-1.756-1.332-1.756-1.09-.746.082-.73.082-.73 1.205.084 1.84 1.236 1.84 1.236 1.07 1.835 2.807 1.305 3.492.998.108-.774.418-1.305.762-1.605-2.665-.304-5.466-1.332-5.466-5.93 0-1.31.468-2.38 1.235-3.22-.124-.304-.535-1.526.117-3.176 0 0 1.008-.322 3.3 1.23a11.5 11.5 0 0 1 3.003-.404 11.5 11.5 0 0 1 3.003.404c2.29-1.552 3.297-1.23 3.297-1.23.653 1.65.242 2.872.119 3.176.77.84 1.233 1.91 1.233 3.22 0 4.61-2.804 5.624-5.476 5.922.43.372.814 1.102.814 2.222 0 1.604-.015 2.895-.015 3.286 0 .322.216.7.826.58C20.565 21.797 24 17.298 24 12 24 5.37 18.63 0 12 0Z" />
            </svg>
          </Link>
        </div>
      </div>
    </footer>
  )
}
