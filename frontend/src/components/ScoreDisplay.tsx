import { useEffect, useState } from 'react'

import { CheckIcon, CrossIcon } from './Icons'
import { Tooltip } from './Tooltip'

/**
 * ScoreDisplay.tsx -- the headline compliance score with its confidence metric.
 *
 * The score counts up on mount. That is not decoration: the number is the single
 * most important thing on the page, and a brief animation draws the eye to it
 * before the reader starts scanning the tables below.
 *
 * Colour is used only for the verdict, and always alongside a glyph and a word,
 * so the result never depends on colour perception alone.
 */

function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

function useCountUp(target: number, durationMs = 750) {
  // Start at the target when motion is reduced, so no animation ever runs and
  // the number is correct on the very first paint.
  const [value, setValue] = useState(() => (prefersReducedMotion() ? target : 0))

  useEffect(() => {
    if (prefersReducedMotion()) return

    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      const progress = Math.min((now - start) / durationMs, 1)
      // Ease-out cubic: fast initial movement, gentle settle.
      setValue(target * (1 - (1 - progress) ** 3))
      if (progress < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, durationMs])

  return prefersReducedMotion() ? target : value
}

/** Thin arc gauge. Monochrome track, ink-coloured progress. */
function ScoreArc({ score, size = 168 }: { score: number; size?: number }) {
  const stroke = 5
  const radius = (size - stroke) / 2
  // A 260-degree sweep leaves a visible gap, which reads as a gauge rather than
  // as a pie chart.
  const sweep = 260
  const circumference = 2 * Math.PI * radius
  const arcLength = (sweep / 360) * circumference
  // A dash array starts at 3 o'clock and runs clockwise. Rotating back by
  // 90 + sweep/2 centres the arc on 12 o'clock, putting the gap at the bottom
  // and starting the fill at the lower left.
  const startAngle = -(90 + sweep / 2)

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      style={{ transform: `rotate(${startAngle}deg)` }}
      aria-hidden="true"
    >
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="rgb(var(--line))"
        strokeWidth={stroke}
        strokeDasharray={`${arcLength} ${circumference}`}
        strokeLinecap="round"
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="rgb(var(--accent))"
        strokeWidth={stroke}
        strokeDasharray={`${(score / 100) * arcLength} ${circumference}`}
        strokeLinecap="round"
        style={{ transition: 'stroke-dasharray 0.75s cubic-bezier(0.32,0.72,0,1)' }}
      />
    </svg>
  )
}

export function ScoreDisplay({
  score,
  confidence,
  confidenceLabel,
  confidenceNote,
  verdict,
}: {
  score: number
  confidence: number
  confidenceLabel: string
  confidenceNote: string
  verdict: 'PASS' | 'FAIL'
}) {
  const animated = useCountUp(score)
  const passed = verdict === 'PASS'

  return (
    <div className="flex flex-col items-center gap-7 sm:flex-row sm:items-center sm:gap-10">
      <div className="relative shrink-0">
        <ScoreArc score={animated} />
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-semibold tabular leading-none tracking-tight text-ink">
            {Math.round(animated)}
          </span>
          <span className="mt-1 text-xs text-ink-faint">out of 100</span>
        </div>
      </div>

      <div className="min-w-0 flex-1 text-center sm:text-left">
        <div
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1
            ${
              passed
                ? 'border-pass/30 bg-pass/10 text-pass'
                : 'border-fail/30 bg-fail/10 text-fail'
            }`}
        >
          {passed ? (
            <CheckIcon className="h-3.5 w-3.5" />
          ) : (
            <CrossIcon className="h-3.5 w-3.5" />
          )}
          <span className="text-xs font-semibold uppercase tracking-label">
            {passed ? 'Predicted Pass' : 'Predicted Fail'}
          </span>
        </div>

        <h2 className="mt-4 text-2xl font-semibold tracking-tight text-ink">
          Compliance score
        </h2>
        <p className="mt-1.5 max-w-md text-sm leading-relaxed text-ink-muted">
          {passed
            ? 'Every band is predicted to sit below the assumed limit line. The score reflects how much headroom remains.'
            : 'At least one band is predicted to exceed the assumed limit line. The score reflects how far over it sits.'}
        </p>

        <div className="mt-5 flex items-center justify-center gap-2 sm:justify-start">
          <span className="text-lg font-semibold tabular text-ink">
            {Math.round(confidence)}
            <span className="text-sm font-normal text-ink-faint">/100</span>
          </span>
          <span className="h-3 w-px bg-line" />
          <span className="text-sm text-ink-muted">{confidenceLabel}</span>
          <Tooltip content={confidenceNote} label="About the confidence score" />
        </div>
      </div>
    </div>
  )
}
