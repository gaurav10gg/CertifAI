import type { Countermeasure } from '../api/types'
import { Eyebrow } from './Card'

/**
 * CountermeasureList.tsx -- ranked design levers, not a single heuristic line.
 */

function formatValue(value: number | string): string {
  if (typeof value === 'string') return value
  return value.toLocaleString(undefined, {
    maximumFractionDigits: Math.abs(value) >= 100 ? 0 : 2,
  })
}

export function CountermeasureList({
  measures,
}: {
  measures: Countermeasure[]
}) {
  if (!measures.length) return null

  return (
    <div>
      <Eyebrow>What to change next</Eyebrow>
      <p className="mt-1 mb-4 text-sm text-ink-muted">
        Ranked by remaining headroom to improve, excluding parameters already
        near their safe limit.
      </p>
      <ol className="space-y-3">
        {measures.map((item, index) => (
          <li
            key={`${item.parameter}-${index}`}
            className="rounded-lg border border-line bg-surface p-4"
          >
            <div className="flex items-baseline justify-between gap-3">
              <p className="text-sm font-medium text-ink">
                <span className="mr-2 tabular text-ink-faint">
                  {String(index + 1).padStart(2, '0')}
                </span>
                {item.label}
              </p>
              <p className="shrink-0 text-2xs tabular text-ink-faint">
                {formatValue(item.current_value)} → {formatValue(item.suggested_value)}
              </p>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">
              {item.suggestion}
            </p>
          </li>
        ))}
      </ol>
    </div>
  )
}
