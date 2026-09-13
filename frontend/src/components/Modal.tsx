import { useCallback, useEffect, useRef, type ReactNode } from 'react'

import { CloseIcon } from './Icons'

/**
 * Modal.tsx -- a full-height right-hand sheet.
 *
 * Handles the accessibility work a dialog needs and is easy to get wrong: Escape
 * to close, a focus trap across Tab and Shift+Tab, focus restored to the trigger
 * on close, background scroll locked, and the panel labelled by its own heading.
 */

export function Modal({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
}) {
  const panelRef = useRef<HTMLDivElement>(null)
  const previouslyFocused = useRef<HTMLElement | null>(null)

  const focusableWithin = useCallback((): HTMLElement[] => {
    if (!panelRef.current) return []
    return Array.from(
      panelRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    ).filter((element) => element.offsetParent !== null)
  }, [])

  useEffect(() => {
    if (!open) return

    previouslyFocused.current = document.activeElement as HTMLElement | null
    const { overflow } = document.body.style
    document.body.style.overflow = 'hidden'

    // Move focus into the sheet so keyboard users are not left behind it.
    const timer = window.setTimeout(() => {
      const [first] = focusableWithin()
      ;(first ?? panelRef.current)?.focus()
    }, 0)

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }
      if (event.key !== 'Tab') return

      const focusable = focusableWithin()
      if (focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = overflow
      window.clearTimeout(timer)
      previouslyFocused.current?.focus()
    }
  }, [open, onClose, focusableWithin])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button
        type="button"
        aria-label="Close"
        onClick={onClose}
        className="absolute inset-0 animate-fade-in bg-ink/25 backdrop-blur-[2px]"
      />

      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        className="relative flex h-full w-full max-w-2xl flex-col border-l
          border-line bg-paper shadow-lifted
          motion-safe:animate-[fade-up_0.28s_cubic-bezier(0.32,0.72,0,1)_both]"
      >
        <header className="flex shrink-0 items-center justify-between gap-4 border-b border-line px-6 py-4">
          <h2 className="text-base font-semibold tracking-tight text-ink">
            {title}
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close panel"
            className="btn-ghost -mr-1 rounded-md p-1.5"
          >
            <CloseIcon className="h-4 w-4" />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-6">{children}</div>
      </div>
    </div>
  )
}
