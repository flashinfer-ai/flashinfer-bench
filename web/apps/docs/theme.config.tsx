import React from 'react'
import type { DocsThemeConfig } from 'nextra-theme-docs'
import { Logo } from '@flashinfer-bench/ui/brand/Logo'
import { links, siteName } from '@flashinfer-bench/config'

const config: DocsThemeConfig = {
  logo: <Logo />,
  project: {
    link: links.siteRepo,
  },
  docsRepositoryBase: links.docsRepositoryBase,
  useNextSeoProps() {
    return {
      titleTemplate: `%s – ${siteName}`,
    }
  },
  footer: {
    text: `${siteName} Documentation`,
  },
  sidebar: {
    defaultMenuCollapseLevel: 1,
  },
}

export default config
