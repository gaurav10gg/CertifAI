/**
 * Disclaimer.tsx -- the standing methodology disclosure.
 *
 * Styled as an ordinary footnote, not a warning banner. The distinction is
 * intentional: this is a permanent, accurate statement of what the tool is, and
 * dressing it as an alert would make users learn to dismiss it. It stays visible
 * on every results view.
 */

export function Disclaimer({
  text,
  onOpenMethodology,
  className = '',
}: {
  text: string
  onOpenMethodology?: () => void
  className?: string
}) {
  return (
    <p className={`text-xs leading-relaxed text-ink-faint ${className}`}>
      {text}
      {onOpenMethodology ? (
        <>
          {' '}
          <button
            type="button"
            onClick={onOpenMethodology}
            className="rounded underline decoration-line-strong underline-offset-2
              transition-colors hover:text-ink-muted hover:decoration-ink-muted"
          >
            Read the methodology
          </button>
          .
        </>
      ) : null}
    </p>
  )
}
