import type { BandResult } from '../api/types'
import { CheckIcon, CrossIcon } from './Icons'

/**
 * BandBreakdown.tsx -- per-band margin to the assumed limit.
 *
 * Labels are Clear / Exceeds rather than pass / fail: this table is a risk
 * detail against a synthetic curve, not a certification verdict.
 *
 * The bar is centred on the limit line: headroom extends right, exceedance
 * extends left. That makes "how close is this to the assumed limit" readable at
 * a glance, which a plain number column does not achieve.
 *
 * Both the model's prediction and the simulator's own measurement are shown. They
 * normally agree to a fraction of a dB; showing both means a disagreement is
 * visible to the user rather than hidden.
 */

/** dB of margin represented by a full half-bar. */
const BAR_FULL_SCALE_DB = 24

function MarginBar({ marginDb }: { marginDb: number }) {
  const magnitude = Math.min(Math.abs(marginDb) / BAR_FULL_SCALE_DB, 1) * 50
  const passes = marginDb > 0

  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-surface">
      {/* Limit line: the centre of the scale. */}
      <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-line-strong" />
      <div
        className={`absolute top-0 h-full ${passes ? 'bg-pass' : 'bg-fail'}`}
        style={
          passes
            ? { left: '50%', width: `${magnitude}%` }
            : { right: '50%', width: `${magnitude}%` }
        }
      />
    </div>
  )
}

function formatHz(hz: number): string {
  if (hz >= 1e6) return `${(hz / 1e6).toFixed(2)} MHz`
  return `${(hz / 1e3).toFixed(0)} kHz`
}

export function BandBreakdown({
  bands,
  unit = 'dBµV',
  showUncertainty = true,
}: {
  bands: BandResult[]
  unit?: string
  showUncertainty?: boolean
}) {
  return (
    <div className="divide-y divide-line">
      {/* Column headers, hidden on small screens where the layout stacks. */}
      <div className="hidden pb-2 sm:grid sm:grid-cols-[1.4fr_0.9fr_0.9fr_1.6fr_auto] sm:items-center sm:gap-4">
        <span className="label-eyebrow">Band</span>
        <span className="label-eyebrow text-right">Peak</span>
        <span className="label-eyebrow text-right">Limit</span>
        <span className="label-eyebrow">Margin to limit</span>
        <span className="label-eyebrow text-right">Result</span>
      </div>

      {bands.map((band) => {
        const passes = band.passes
        const agrees = passes === (band.simulated_margin_db > 0)

        return (
          <div
            key={band.key}
            className="grid grid-cols-2 gap-x-4 gap-y-2 py-3.5
              sm:grid-cols-[1.4fr_0.9fr_0.9fr_1.6fr_auto] sm:items-center"
          >
            <div className="col-span-2 sm:col-span-1">
              <p className="text-sm font-medium text-ink">{band.label}</p>
              <p className="mt-0.5 text-2xs tabular text-ink-faint">
                Worst at {formatHz(band.worst_frequency_hz)}
                {band.harmonic_count > 0
                  ? ` · ${band.harmonic_count} near-limit lines`
                  : ''}
              </p>
            </div>

            <p className="text-sm tabular text-ink sm:text-right">
              <span className="text-2xs text-ink-faint sm:hidden">Peak </span>
              {band.simulated_peak_dbuv.toFixed(1)}
              <span className="text-2xs text-ink-faint"> {unit}</span>
            </p>

            <p className="text-sm tabular text-ink-muted sm:text-right">
              <span className="text-2xs text-ink-faint sm:hidden">Limit </span>
              {band.limit_at_peak_dbuv.toFixed(1)}
              <span className="text-2xs text-ink-faint"> {unit}</span>
            </p>

            <div className="col-span-2 sm:col-span-1">
              <div className="flex items-center gap-3">
                <MarginBar marginDb={band.predicted_margin_db} />
                <span
                  className={`w-20 shrink-0 text-right text-sm font-medium tabular ${
                    passes ? 'text-ink' : 'text-fail'
                  }`}
                >
                  {band.predicted_margin_db >= 0 ? '+' : ''}
                  {band.predicted_margin_db.toFixed(1)} dB
                </span>
              </div>
              {showUncertainty ? (
                <p className="mt-1 text-2xs tabular text-ink-faint">
                  ±{band.margin_uncertainty_db.toFixed(1)} dB model error · simulated{' '}
                  {band.simulated_margin_db >= 0 ? '+' : ''}
                  {band.simulated_margin_db.toFixed(1)} dB
                  {agrees ? '' : ' · prediction and simulation disagree'}
                </p>
              ) : (
                <p className="mt-1 text-2xs tabular text-ink-faint">
                  Simulated margin {band.simulated_margin_db >= 0 ? '+' : ''}
                  {band.simulated_margin_db.toFixed(1)} dB
                </p>
              )}
            </div>

            <div className="col-span-2 flex sm:col-span-1 sm:justify-end">
              <span
                className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1
                  text-2xs font-semibold uppercase tracking-label
                  ${
                    passes
                      ? 'border-pass/30 bg-pass/10 text-pass'
                      : 'border-fail/30 bg-fail/10 text-fail'
                  }`}
              >
                {passes ? (
                  <CheckIcon className="h-3 w-3" />
                ) : (
                  <CrossIcon className="h-3 w-3" />
                )}
                {passes ? 'Clear' : 'Exceeds'}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
