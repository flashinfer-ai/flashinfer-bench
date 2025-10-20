import { SiteHeader } from "@flashinfer-bench/ui"
import { docsBasePath } from "@flashinfer-bench/config"

const NAV_ITEMS = [
  { href: docsBasePath, label: "Docs", external: true },
  { href: "/viewer", label: "Viewer" },
]

export function Header() {
  return <SiteHeader navItems={NAV_ITEMS} />
}
