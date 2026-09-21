import type { AssessmentHistoryEntry } from '../api/types'
import { Eyebrow } from './Card'
import { riskBadgeClass } from './ScoreDisplay'

/**
 * HistorySparkline.tsx -- last three assessments in this session.
 */

export function HistorySparkline({
  history,
}: {
  history: AssessmentHistoryEntry[]
}) {
  if (history.length < 2) return null

  const scores = [...history].reverse().map((entry) => entry.score)
  const min = 0
  const max = 100
  const width = 160
  const height = 36
  const pad = 3
  const points = scores.map((score, index) => {
    const x =
      scores.length === 1
        ? width / 2
        : pad + (index / (scores.length - 1)) * (width - pad * 2)
    const y = pad + (1 - (score - min) / (max - min)) * (height - pad * 2)
    return `${x},${y}`
  })

  return (
    <div className="rounded-card border border-line bg-surface p-4">
      <div className="flex items-center justify-between gap-3">
        <Eyebrow>What changed</Eyebrow>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-8 w-36 text-ink"
          aria-hidden="true"
        >
          <polyline
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            points={points.join(' ')}
          />
          {scores.map((score, index) => {
            const [x, y] = points[index].split(',').map(Number)
            return (
              <circle
                key={`${score}-${index}`}
                cx={x}
                cy={y}
                r="2.2"
                fill="currentColor"
              />
            )
          })}
        </svg>
      </div>
      <ol className="mt-3 flex flex-wrap gap-3">
        {[...history].reverse().map((entry, index) => (
          <li key={`${entry.at}-${index}`} className="flex items-center gap-2">
            <span
              className={`rounded-full border px-2 py-0.5 text-2xs font-semibold uppercase tracking-label ${riskBadgeClass(entry.level)}`}
            >
              {entry.level}
            </span>
            <span className="text-xs tabular text-ink">
              {entry.score.toFixed(0)} ±{entry.plusMinus.toFixed(0)}
            </span>
            <span className="text-2xs text-ink-faint">{entry.name}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
