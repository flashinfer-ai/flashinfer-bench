import nextra from 'nextra'

const withNextra = nextra({
})

export default withNextra({
  reactStrictMode: true,
  basePath: '/docs',
  transpilePackages: [
    '@flashinfer-bench/ui',
    '@flashinfer-bench/utils',
    '@flashinfer-bench/config',
  ],
})
