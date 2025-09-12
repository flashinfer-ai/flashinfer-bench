import React from 'react'
import type { DocsThemeConfig } from 'nextra-theme-docs'

const config: DocsThemeConfig = {
  logo: <span>FlashInfer Bench Docs</span>,
  project: {
    link: 'https://github.com/flashinfer-ai/flashinfer-bench',
  },
  docsRepositoryBase:
    'https://github.com/flashinfer-ai/flashinfer-bench/tree/main/docs',
  useNextSeoProps() {
    return {
      titleTemplate: '%s – FlashInfer Bench',
    }
  },
  footer: {
    text: 'FlashInfer Bench Documentation',
  },
  sidebar: {
    defaultMenuCollapseLevel: 1,
  },
}

export default config
