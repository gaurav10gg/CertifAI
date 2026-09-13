import type { RiskFactor } from '../api/types'
import { Eyebrow } from './Card'
import { WaveIcon } from './Icons'

/**
 * RiskCallout.tsx -- names the parameter most responsible for the outcome and
 * proposes a concrete change.
 *
 * The attribution comes from exact tree SHAP values on the worst band's
 * classifier (see backend/predictor.py), weighted by how far the parameter
 * already sits towards its own risky extreme -- so a parameter already at its
 * safest setting is not suggested for further change.
 *
 * Marked with a heavy left rule rather than a coloured banner: this is guidance,
 * not an alarm.
 */

function formatValue(value: number | string, unit: string): string {
  if (typeof value === 'string') return value
  const formatted = value.toLocaleString(undefined, {
    maximumFractionDigits: Math.abs(value) >= 100 ? 0 : 1,
  })
  return unit ? `${formatted} ${unit}` : formatted
}

export function RiskCallout({ risk }: { risk: RiskFactor }) {
  return (
    <section className="rounded-card border border-line border-l-2 border-l-ink bg-surface p-5">
      <div className="flex items-center gap-2">
        <WaveIcon className="h-3.5 w-3.5 text-ink-muted" />
        <Eyebrow>Top risk factor</Eyebrow>
      </div>

      <p className="mt-2.5 text-base font-medium leading-snug text-ink">
        {risk.statement}
      </p>
      <p className="mt-2.5 max-w-2xl text-sm leading-relaxed text-ink-muted">
        {risk.suggestion}
      </p>

      <dl className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-line pt-3.5">
        <div className="flex items-baseline gap-2">
          <dt className="text-2xs uppercase tracking-label text-ink-faint">
            Current
          </dt>
          <dd className="text-sm tabular text-ink">
            {formatValue(risk.current_value, risk.unit)}
          </dd>
        </div>
        <div className="flex items-baseline gap-2">
          <dt className="text-2xs uppercase tracking-label text-ink-faint">
            Suggested
          </dt>
          <dd className="text-sm font-medium tabular text-ink">
            {formatValue(risk.suggested_value, risk.unit)}
          </dd>
        </div>
        <div className="flex items-baseline gap-2">
          <dt className="text-2xs uppercase tracking-label text-ink-faint">
            Driving band
          </dt>
          <dd className="text-sm text-ink-muted">{risk.driving_band_label}</dd>
        </div>
      </dl>
    </section>
  )
}
