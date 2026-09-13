import { MoonIcon, SunIcon } from './Icons'
import type { ThemeMode } from '../theme/useTheme'

/**
 * Header.tsx -- fixed application header: wordmark, methodology link, theme
 * toggle.
 *
 * The wordmark doubles as a home action once an assessment is under way. The
 * "Simulation" tag next to it is a permanent, quiet reminder of what this tool
 * is, placed where a product would normally put a version badge.
 */

export function Header({
  theme,
  onToggleTheme,
  onOpenMethodology,
  onReset,
  canReset,
}: {
  theme: ThemeMode
  onToggleTheme: () => void
  onOpenMethodology: () => void
  onReset: () => void
  canReset: boolean
}) {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-paper/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-5 sm:px-8">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={canReset ? onReset : undefined}
            disabled={!canReset}
            className="group flex items-center gap-2.5 rounded disabled:cursor-default"
            aria-label={canReset ? 'Start a new assessment' : 'CertifAI'}
          >
            <Wordmark />
          </button>
          <span
            className="hidden rounded border border-line px-1.5 py-0.5 text-2xs
              font-medium uppercase tracking-label text-ink-faint sm:inline"
          >
            Simulation
          </span>
        </div>

        <nav className="flex items-center gap-1">
          <button
            type="button"
            onClick={onOpenMethodology}
            className="btn-ghost text-sm"
          >
            Methodology
          </button>
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label={
              theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
            }
            className="btn-ghost rounded-md p-2"
          >
            {theme === 'dark' ? (
              <SunIcon className="h-4 w-4" />
            ) : (
              <MoonIcon className="h-4 w-4" />
            )}
          </button>
        </nav>
      </div>
    </header>
  )
}

/**
 * The wordmark. The glyph is a stylised spectrum trace crossing a limit line,
 * which is the product's core idea reduced to five strokes.
 */
function Wordmark() {
  return (
    <span className="flex items-center gap-2">
      <svg
        viewBox="0 0 22 22"
        className="h-[22px] w-[22px] text-ink"
        fill="none"
        aria-hidden="true"
      >
        <rect
          x="0.75"
          y="0.75"
          width="20.5"
          height="20.5"
          rx="5"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <path
          d="M4.5 13.5h1.6l1.4-4.2 1.9 6.2 1.8-8 1.6 5.4 1.3-2.4h3.4"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M4.5 6.6h13"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeDasharray="2.2 2"
          strokeLinecap="round"
          opacity="0.45"
        />
      </svg>
      <span className="text-sm font-semibold tracking-tight text-ink">
        CertifAI
      </span>
    </span>
  )
}
