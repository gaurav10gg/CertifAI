import { useEffect, useState } from 'react'

import type { RiskLevel } from '../api/types'
import { Tooltip } from './Tooltip'

/**
 * ScoreDisplay.tsx -- headline risk score with ensemble uncertainty.
 *
 * The score counts up on mount. Colour is used only as a secondary cue beside
 * the tier label, so the result never depends on colour perception alone.
 */

function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

function useCountUp(target: number, durationMs = 750) {
  const [value, setValue] = useState(() => (prefersReducedMotion() ? target : 0))

  useEffect(() => {
    if (prefersReducedMotion()) return

    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      const progress = Math.min((now - start) / durationMs, 1)
      setValue(target * (1 - (1 - progress) ** 3))
      if (progress < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, durationMs])

  return prefersReducedMotion() ? target : value
}

function ScoreArc({ score, size = 168 }: { score: number; size?: number }) {
  const stroke = 5
  const radius = (size - stroke) / 2
  const sweep = 260
  const circumference = 2 * Math.PI * radius
  const arcLength = (sweep / 360) * circumference
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

export function riskBadgeClass(level: RiskLevel | string): string {
  if (level === 'HIGH') return 'border-fail/30 bg-fail/10 text-fail'
  if (level === 'LOW') return 'border-pass/30 bg-pass/10 text-pass'
  return 'border-line-strong bg-surface text-ink'
}

export function ScoreDisplay({
  score,
  plusMinus,
  scoreLow,
  scoreHigh,
  riskLevel,
  riskLabel,
  riskCopy,
  confidence,
  confidenceLabel,
  confidenceNote,
}: {
  score: number
  plusMinus: number
  scoreLow: number
  scoreHigh: number
  riskLevel: RiskLevel
  riskLabel: string
  riskCopy: string
  confidence: number
  confidenceLabel: string
  confidenceNote: string
}) {
  const animated = useCountUp(score)

  return (
    <div className="flex flex-col items-center gap-7 sm:flex-row sm:items-center sm:gap-10">
      <div className="relative shrink-0">
        <ScoreArc score={animated} />
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-semibold tabular leading-none tracking-tight text-ink">
            {Math.round(animated)}
          </span>
          <span className="mt-1 text-xs tabular text-ink-faint">
            ±{plusMinus.toFixed(0)}
          </span>
        </div>
      </div>

      <div className="min-w-0 flex-1 text-center sm:text-left">
        <div
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 ${riskBadgeClass(riskLevel)}`}
        >
          <span className="text-xs font-semibold uppercase tracking-label">
            {riskLabel}
          </span>
        </div>

        <h2 className="mt-4 text-2xl font-semibold tracking-tight text-ink">
          Risk score {Math.round(score)}
          <span className="text-lg font-normal text-ink-muted">
            {' '}
            (±{plusMinus.toFixed(0)})
          </span>
        </h2>
        <p className="mt-1.5 max-w-md text-sm leading-relaxed text-ink-muted">
          {riskCopy} Ensemble range {scoreLow.toFixed(0)}–{scoreHigh.toFixed(0)}.
          Higher score means more simulated headroom, not a lab pass.
        </p>

        <div className="mt-5 flex items-center justify-center gap-2 sm:justify-start">
          <span className="text-lg font-semibold tabular text-ink">
            {Math.round(confidence)}
            <span className="text-sm font-normal text-ink-faint">/100</span>
          </span>
          <span className="h-3 w-px bg-line" />
          <span className="text-sm text-ink-muted">{confidenceLabel}</span>
          <Tooltip content={confidenceNote} label="About the uncertainty band" />
        </div>
      </div>
    </div>
  )
}
