/**
 * Tailwind configuration for CertifAI.
 *
 * Every colour is a semantic token resolving to a CSS custom property defined in
 * src/theme/tokens.css. Components therefore never name a literal colour, and
 * dark mode is a single class toggle on <html> rather than a `dark:` variant on
 * every element.
 *
 * The palette is strictly monochrome. `pass` and `fail` are the only chromatic
 * tokens and are reserved for compliance verdicts.
 */

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: 'rgb(var(--paper) / <alpha-value>)',
        surface: 'rgb(var(--surface) / <alpha-value>)',
        raised: 'rgb(var(--raised) / <alpha-value>)',
        ink: 'rgb(var(--ink) / <alpha-value>)',
        'ink-muted': 'rgb(var(--ink-muted) / <alpha-value>)',
        'ink-faint': 'rgb(var(--ink-faint) / <alpha-value>)',
        line: 'rgb(var(--line) / <alpha-value>)',
        'line-strong': 'rgb(var(--line-strong) / <alpha-value>)',
        accent: 'rgb(var(--accent) / <alpha-value>)',
        'accent-ink': 'rgb(var(--accent-ink) / <alpha-value>)',
        pass: 'rgb(var(--pass) / <alpha-value>)',
        fail: 'rgb(var(--fail) / <alpha-value>)',
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'sans-serif',
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          'monospace',
        ],
      },
      fontSize: {
        // A deliberately small type scale; engineering tools read better with
        // fewer, more consistent sizes.
        '2xs': ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.04em' }],
        display: ['4.5rem', { lineHeight: '1', letterSpacing: '-0.035em' }],
      },
      letterSpacing: {
        label: '0.09em',
      },
      borderRadius: {
        card: '0.625rem',
      },
      boxShadow: {
        // Subtle elevation only -- no dramatic drop shadows anywhere.
        subtle: '0 1px 2px 0 rgb(0 0 0 / 0.04)',
        lifted: '0 4px 16px -4px rgb(0 0 0 / 0.10)',
      },
      transitionTimingFunction: {
        standard: 'cubic-bezier(0.32, 0.72, 0, 1)',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        sweep: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.32s cubic-bezier(0.32, 0.72, 0, 1) both',
        'fade-in': 'fade-in 0.2s ease-out both',
        sweep: 'sweep 1.4s cubic-bezier(0.4, 0, 0.2, 1) infinite',
      },
    },
  },
  plugins: [],
}
