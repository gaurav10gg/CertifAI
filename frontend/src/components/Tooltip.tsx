import { useId, useState } from 'react'

import { InfoIcon } from './Icons'

/**
 * Tooltip.tsx -- an info affordance that explains what a parameter physically is.
 *
 * Opens on hover *and* on focus, and is linked to its trigger by
 * aria-describedby, so the explanation is reachable by keyboard and to a screen
 * reader rather than being hover-only decoration.
 */

export function Tooltip({
  content,
  label,
}: {
  content: string
  /** Accessible name for the trigger, e.g. "About Switching Frequency". */
  label: string
}) {
  const [open, setOpen] = useState(false)
  const id = useId()

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((value) => !value)}
        className="rounded text-ink-faint transition-colors hover:text-ink"
      >
        <InfoIcon className="h-3.5 w-3.5" />
      </button>

      {open ? (
        <span
          id={id}
          role="tooltip"
          className="absolute bottom-full left-1/2 z-30 mb-2 w-64 -translate-x-1/2
            animate-fade-in rounded-lg border border-line bg-raised p-3
            text-xs leading-relaxed text-ink-muted shadow-lifted"
        >
          {content}
        </span>
      ) : null}
    </span>
  )
}
