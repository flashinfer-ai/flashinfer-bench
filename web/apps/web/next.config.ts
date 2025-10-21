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

      // Sphinx documentation
      { source: '/docs/api',                destination: '/pyapi/index.html' },
      { source: '/docs/api/python',         destination: '/pyapi/index.html' },
      { source: '/docs/api/:path*',         destination: '/pyapi/:path*' },

      // Mintlify documentation
      { source: '/docs', destination: `${DOCS_ORIGIN}/docs` },
      { source: '/docs/:path*', destination: `${DOCS_ORIGIN}/docs/:path*` },
      // Mintlify assets
      { source: '/mintlify-assets/:path*', destination: `${DOCS_ORIGIN}/mintlify-assets/:path*` },
      { source: '/_mintlify/:path*', destination: `${DOCS_ORIGIN}/_mintlify/:path*` },
      // Mintlify next assets
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
    ]
  },
  async headers() {
    return [
      {
        source: '/pyapi/:all*(css|js|png|jpg|gif|svg|ico|woff|woff2)',
        headers: [{ key: 'Cache-Control', value: 'public, max-age=31536000, immutable' }],
      },
      {
        source: '/pyapi/:path*',
        headers: [{ key: 'Cache-Control', value: 'public, max-age=60' }],
      },
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
