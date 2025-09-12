import nextra from 'nextra'

const withNextra = nextra({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.tsx',
  latex: false,
  mdxOptions: {},
})

export default withNextra({
  basePath: '/docs',
  reactStrictMode: true,
  // Include TS/JS so Next can see _app.tsx alongside MD/MDX content.
  pageExtensions: ['ts', 'tsx', 'js', 'jsx', 'md', 'mdx'],
  experimental: {
    // Allows importing from outside the app root if needed.
    externalDir: true,
  }
})
