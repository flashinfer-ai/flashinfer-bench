/** @type {import('next').NextConfig} */
const DOCS_ORIGIN = process.env.DOCS_ORIGIN || 'http://localhost:3030'
const nextConfig = {
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
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on'
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN'
          },
        ],
      },
    ]
  },
}

module.exports = nextConfig
