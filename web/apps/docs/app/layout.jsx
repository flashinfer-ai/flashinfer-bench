import './globals.css'

import { Inter } from 'next/font/google'
import { Layout } from 'nextra-theme-docs'
import { links } from '@flashinfer-bench/config'
import { Head, Search } from 'nextra/components'
import { SiteFooter, SiteHeader } from '@flashinfer-bench/ui'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'

export const metadata = {
  // Define your metadata here
  // For more information on metadata API, see: https://nextjs.org/docs/app/building-your-application/optimizing/metadata
}

const APP_HOME = process.env.NEXT_PUBLIC_APP_HOME ?? 'https://bench.flashinfer.ai'

const inter = Inter({ subsets: ['latin'] })

const navbar = (
  <SiteHeader
    logoHref={APP_HOME}
    navItems={[]}
    searchSlot={<Search />}
  />
)
export default async function RootLayout({ children }) {
  return (
    <html
      // Not required, but good for SEO
      lang="en"
      // Required to be set
      dir="ltr"
      // Suggested by `next-themes` package https://github.com/pacocoursey/next-themes#with-app
      suppressHydrationWarning
    >
      <Head
      // ... Your additional head options
      >
        {/* Your additional tags should be passed as `children` of `<Head>` element */}
      </Head>
      <body className={inter.className}>
        <Layout
          navbar={navbar}
          pageMap={await getPageMap()}
          docsRepositoryBase={links.docsRepositoryBase}
          footer={null}
          // ... Your additional layout options
        >
          {children}
        </Layout>
        <SiteFooter />
      </body>
    </html>
  )
}
