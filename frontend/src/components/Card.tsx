import type { ReactNode } from 'react'

/**
 * Card.tsx -- the surface primitive.
 *
 * Two variants: a plain panel, and a selectable card used for the device grid.
 * Selection is expressed by border weight and a corner marker rather than by
 * colour, keeping the monochrome constraint intact.
 */

type CardProps = {
  children: ReactNode
  className?: string
  /** Renders the card as a padded panel. Set false to control padding yourself. */
  padded?: boolean
}

export function Card({ children, className = '', padded = true }: CardProps) {
  return (
    <div className={`card-base ${padded ? 'p-5' : ''} ${className}`}>
      {children}
    </div>
  )
}

type SelectableCardProps = {
  children: ReactNode
  selected?: boolean
  onClick: () => void
  className?: string
  ariaLabel?: string
}

export function SelectableCard({
  children,
  selected = false,
  onClick,
  className = '',
  ariaLabel,
}: SelectableCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      aria-label={ariaLabel}
      className={`group relative flex w-full flex-col items-start rounded-card
        border bg-raised p-5 text-left transition-all duration-200 ease-standard
        hover:-translate-y-0.5 hover:shadow-lifted
        ${
          selected
            ? 'border-accent ring-1 ring-accent'
            : 'border-line hover:border-line-strong'
        }
        ${className}`}
    >
      {children}
    </button>
  )
}

/** Small uppercase label used above groups of content. */
export function Eyebrow({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return <p className={`label-eyebrow ${className}`}>{children}</p>
}

/** A labelled statistic. Numbers use tabular figures so columns line up. */
export function Stat({
  label,
  value,
  suffix,
  hint,
  className = '',
}: {
  label: string
  value: ReactNode
  suffix?: string
  hint?: string
  className?: string
}) {
  return (
    <div className={className}>
      <Eyebrow>{label}</Eyebrow>
      <p className="mt-1.5 text-xl font-semibold tabular text-ink">
        {value}
        {suffix ? (
          <span className="ml-1 text-sm font-normal text-ink-faint">
            {suffix}
          </span>
        ) : null}
      </p>
      {hint ? <p className="mt-1 text-xs text-ink-muted">{hint}</p> : null}
    </div>
  )
}
