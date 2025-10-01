import type { NextConfig } from 'next'
import { withMicrofrontends } from '@vercel/microfrontends/next/config'
import nextra from 'nextra'

const withNextra = nextra({})

const config: NextConfig = {
  reactStrictMode: true,
  basePath: '/docs',
  transpilePackages: [
    '@flashinfer-bench/ui',
    '@flashinfer-bench/utils',
    '@flashinfer-bench/config',
  ],
}

export default withMicrofrontends(withNextra(config))
