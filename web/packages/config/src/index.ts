export const siteName = 'FlashInfer-Bench'
export const siteDescription = 'AI for AI Infrastructure for Accelerating AI Deployment'

export const links = {
  org: 'https://github.com/flashinfer-ai',
  siteRepo: 'https://github.com/flashinfer-ai/flashinfer-bench',
  docsRepositoryBase: 'https://github.com/flashinfer-ai/flashinfer-bench/tree/main/docs',
}

export const docsBasePath = '/docs'

export const env = {
  docsOriginVar: 'DOCS_ORIGIN',
}

export function getDefaultMetadata() {
  return {
    title: siteName,
    description: siteDescription,
  }
}
