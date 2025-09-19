import type { StaticImageData } from "next/image"

import fibBlackBg from "./fib-black-bg.png"
import fibWhiteBg from "./fib-white-bg.png"

import { cn } from "@flashinfer-bench/utils"

type LogoVariant = "auto" | "light" | "dark"

export interface LogoProps {
  className?: string
  /**
   * Controls which asset to render. Use `light` for light backgrounds,
   * `dark` for dark backgrounds, or `auto` to follow user preferences.
   */
  variant?: LogoVariant
  /** Accessible label for the logo. */
  alt?: string
  /** Applies directly to the rendered `<img>` element. */
  imgClassName?: string
  /** Pixel height for the image (defaults to 32px). */
  height?: number
}

const toSrc = (image: StaticImageData | string) =>
  typeof image === "string" ? image : image.src

const LIGHT_LOGO_SRC = toSrc(fibWhiteBg)
const DARK_LOGO_SRC = toSrc(fibBlackBg)

export function Logo({
  className,
  variant = "auto",
  alt = "FlashInfer Bench",
  imgClassName,
  height = 36,
}: LogoProps) {
  const wrapperClasses = cn("inline-flex items-center", className)

  const renderImg = (src: string, ariaHidden = false) => (
    <img
      src={src}
      alt={ariaHidden ? "" : alt}
      aria-hidden={ariaHidden || undefined}
      className={cn("block", imgClassName)}
      loading="eager"
      decoding="async"
      style={{ height, width: "auto" }}
    />
  )

  if (variant === "light") {
    return <span className={wrapperClasses}>{renderImg(LIGHT_LOGO_SRC)}</span>
  }

  if (variant === "dark") {
    return <span className={wrapperClasses}>{renderImg(DARK_LOGO_SRC)}</span>
  }

  return (
    <span className={wrapperClasses}>
      <picture>
        <source srcSet={DARK_LOGO_SRC} media="(prefers-color-scheme: dark)" />
        {renderImg(LIGHT_LOGO_SRC)}
      </picture>
    </span>
  )
}
