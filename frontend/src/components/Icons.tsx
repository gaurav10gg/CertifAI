/**
 * Icons.tsx -- inline monochrome SVG icons.
 *
 * All icons are stroke-only, 1.5 px, on a 24-unit grid, and inherit
 * currentColor. Device icons are simple mechanical abstractions rather than
 * literal illustrations, which keeps them legible at card size and consistent
 * with the monochrome system.
 */

import type { ReactElement, ReactNode } from 'react'

type IconProps = {
  className?: string
  strokeWidth?: number
}

function Svg({
  children,
  className = 'h-5 w-5',
  strokeWidth = 1.5,
}: IconProps & { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {children}
    </svg>
  )
}

/* ---------------------------------------------------------------- device icons */

/** Gearless traction machine: sheave with a wrapped rope. */
export function GearlessIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="12" cy="10" r="5.5" />
      <circle cx="12" cy="10" r="1.75" />
      <path d="M6.5 10v7.5a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2V10" />
    </Svg>
  )
}

/** High-rise shaft: tall building with a long travel path. */
export function HighRiseIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M6 21V4.5A1.5 1.5 0 0 1 7.5 3h9A1.5 1.5 0 0 1 18 4.5V21" />
      <path d="M3.5 21h17" />
      <path d="M12 6.5v9" />
      <path d="M9.5 13 12 15.5l2.5-2.5" />
    </Svg>
  )
}

/** Compact controller: small enclosure with terminal rail. */
export function CompactIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <rect x="4" y="5" width="16" height="14" rx="1.75" />
      <path d="M7.5 9h9" />
      <path d="M7.5 12.5h5.5" />
      <path d="M7.5 16h3" />
    </Svg>
  )
}

/** Freight: heavy load platform. */
export function FreightIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3.5 17.5h17" />
      <rect x="6" y="9" width="12" height="8.5" rx="1" />
      <path d="M12 9V4.5" />
      <path d="M9 6.5h6" />
      <path d="M7 20.5h10" />
    </Svg>
  )
}

/** SiC device: semiconductor die with fast-edge waveform. */
export function SicIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <rect x="5" y="7" width="14" height="10" rx="1.5" />
      <path d="M8.5 14.5V11l2.5-.001V14.5l2.5.001V11h2" />
      <path d="M9 4v3M15 4v3M9 17v3M15 17v3" />
    </Svg>
  )
}

/** Escalator: inclined step run. */
export function EscalatorIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3.5 18.5h4l9-11h4" />
      <path d="M6 18.5v-3h3.5v-3H13v-3h3.5" />
      <circle cx="19" cy="6" r="1.25" />
    </Svg>
  )
}

/** Custom configuration: sliders. */
export function CustomIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M5 6h14M5 12h14M5 18h14" />
      <circle cx="9" cy="6" r="2" />
      <circle cx="15" cy="12" r="2" />
      <circle cx="8" cy="18" r="2" />
    </Svg>
  )
}

const DEVICE_ICONS: Record<string, (props: IconProps) => ReactElement> = {
  gearless: GearlessIcon,
  highrise: HighRiseIcon,
  compact: CompactIcon,
  freight: FreightIcon,
  sic: SicIcon,
  escalator: EscalatorIcon,
  custom: CustomIcon,
}

export function DeviceIcon({
  name,
  className,
}: {
  name: string
  className?: string
}) {
  const Resolved = DEVICE_ICONS[name] ?? CompactIcon
  return <Resolved className={className} />
}

/* ---------------------------------------------------------------- ui icons */

export function SunIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M18.8 5.2l-1.4 1.4M6.6 17.4l-1.4 1.4" />
    </Svg>
  )
}

export function MoonIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z" />
    </Svg>
  )
}

export function ArrowRightIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M5 12h14M13 6l6 6-6 6" />
    </Svg>
  )
}

export function ArrowLeftIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M19 12H5M11 18l-6-6 6-6" />
    </Svg>
  )
}

export function DownloadIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 3.5v11M7.5 10.5 12 15l4.5-4.5" />
      <path d="M4.5 17v2.5h15V17" />
    </Svg>
  )
}

export function CheckIcon(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={props.strokeWidth ?? 2}>
      <path d="M4.5 12.5 9.5 17.5 19.5 6.5" />
    </Svg>
  )
}

export function CrossIcon(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={props.strokeWidth ?? 2}>
      <path d="M6 6l12 12M18 6L6 18" />
    </Svg>
  )
}

export function InfoIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5.5" />
      <path d="M12 7.75v.5" />
    </Svg>
  )
}

export function CloseIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" />
    </Svg>
  )
}

export function SpinnerIcon({ className = 'h-4 w-4' }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={`${className} animate-spin`}
      fill="none"
      aria-hidden="true"
    >
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeOpacity="0.2"
      />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function WaveIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M2.5 12h3l2-6 3 12 3-9 2.5 5h5.5" />
    </Svg>
  )
}
