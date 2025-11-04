import type { Metadata } from "next"
import { getDefaultMetadata } from "@flashinfer-bench/config"
import { Inter } from "next/font/google"
import "./globals.css"
import { Providers } from "@flashinfer-bench/ui"
import { Header } from "@/components/layout/header"
import { Footer } from "@/components/layout/footer"
import { Toaster } from "@flashinfer-bench/ui"
import { Analytics } from "@vercel/analytics/next"


const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = getDefaultMetadata() as Metadata

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <div className="relative flex min-h-screen flex-col">
            <Header />
            <main className="flex-1">{children}</main>
            <Footer />
          </div>
          <Toaster />
        </Providers>
        <Analytics />
      </body>
    </html>
  )
}
