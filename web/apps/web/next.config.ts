import type { NextConfig } from 'next'
import { withMicrofrontends } from '@vercel/microfrontends/next/config'

const DOCS_ORIGIN = process.env.DOCS_ORIGIN ?? 'http://localhost:3030'

const nextConfig: NextConfig = {
  transpilePackages: [
    '@flashinfer-bench/ui',
    '@flashinfer-bench/utils',
    '@flashinfer-bench/config',
  ],
  async rewrites() {
    return [
      { source: '/docs', destination: `${DOCS_ORIGIN}/docs` },
      { source: '/docs/:path*', destination: `${DOCS_ORIGIN}/docs/:path*` },
    ]
  },
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-DNS-Prefetch-Control', value: 'on' },
          { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
        ],
      },
    ]
  },
}

export default withMicrofrontends(nextConfig)
