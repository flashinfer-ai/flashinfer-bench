import type { NextConfig } from 'next'
// import { withMicrofrontends } from '@vercel/microfrontends/next/config'

const DOCS_ORIGIN =
  process.env.DOCS_ORIGIN ??
  // Revert to 'http://localhost:3030' when re-enabling the docs microfrontend.
  'https://flashinfer-bench.mintlify.app'

const nextConfig: NextConfig = {
  transpilePackages: [
    '@flashinfer-bench/ui',
    '@flashinfer-bench/utils',
    '@flashinfer-bench/config',
  ],
  async rewrites() {
    return [
      {
        source: '/docs',
        destination: `${DOCS_ORIGIN}/docs`,
      },
      {
        source: '/docs/:path*',
        destination: `${DOCS_ORIGIN}/docs/:path*`,
      },
      // Mintlify assets currently load from the `/mintlify-assets` and `/_mintlify` prefixes.
      // Proxy them while the docs microfrontend is disabled. Remove when reverting.
      {
        source: '/mintlify-assets/:path*',
        destination: `${DOCS_ORIGIN}/mintlify-assets/:path*`,
      },
      {
        source: '/_mintlify/:path*',
        destination: `${DOCS_ORIGIN}/_mintlify/:path*`,
      },
      // Mintlify serves assets from the root `/_next` paths. We proxy them when requested from the docs
      // section to avoid clashing with the main app's assets.
      {
        source: '/_next/static/:path*',
        has: [
          {
            type: 'header',
            key: 'referer',
            value: 'https?://[^/]+/docs(?:/.*)?',
          },
        ],
        destination: `${DOCS_ORIGIN}/_next/static/:path*`,
      },
      {
        source: '/_next/image/:path*',
        has: [
          {
            type: 'header',
            key: 'referer',
            value: 'https?://[^/]+/docs(?:/.*)?',
          },
        ],
        destination: `${DOCS_ORIGIN}/_next/image/:path*`,
      },
      {
        source: '/socket.io',
        destination: `${DOCS_ORIGIN}/socket.io`,
      },
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

// export default withMicrofrontends(nextConfig)
export default nextConfig
