import { useCallback, useEffect, useState } from 'react'

/**
 * Dark/light mode, persisted to localStorage and defaulting to the OS preference.
 *
 * The class is applied to <html> so every token in tokens.css switches at once.
 * An inline script in index.html applies the same class before first paint, so
 * there is no light flash on load for dark-mode users.
 */

export type ThemeMode = 'light' | 'dark'

const STORAGE_KEY = 'certifai-theme'

export function resolveInitialTheme(): ThemeMode {
  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored === 'light' || stored === 'dark') return stored
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light'
}

export function useTheme() {
  const [theme, setTheme] = useState<ThemeMode>(() => {
    // The pre-paint script has already set the class; read it back rather than
    // recomputing, so the two can never disagree.
    return document.documentElement.classList.contains('dark')
      ? 'dark'
      : 'light'
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    localStorage.setItem(STORAGE_KEY, theme)
  }, [theme])

  // Follow the OS only while the user has not made an explicit choice.
  useEffect(() => {
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = (event: MediaQueryListEvent) => {
      if (localStorage.getItem(STORAGE_KEY)) return
      setTheme(event.matches ? 'dark' : 'light')
    }
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])

  const toggle = useCallback(() => {
    setTheme((current) => (current === 'dark' ? 'light' : 'dark'))
  }, [])

  return { theme, toggle }
}

/**
 * Read the current value of a design token.
 *
 * Recharts needs real colour strings rather than CSS variables for its SVG
 * attributes, so chart components resolve tokens through here and re-resolve when
 * the theme changes.
 */
export function readToken(name: string): string {
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim()
  if (!raw) return 'rgb(0 0 0)'
  // Tokens are stored as "10 10 10" channel triplets.
  return /^[\d\s.]+$/.test(raw) ? `rgb(${raw})` : raw
}
