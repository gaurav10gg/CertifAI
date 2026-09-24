import { useMemo, useState } from 'react'

import { ApiError, comparePredictions } from '../api/client'
import type {
  CompareResult,
  DeviceParameters,
  DeviceProfile,
  ParameterSpec,
  PredictionResult,
} from '../api/types'
import { Card, Eyebrow } from './Card'
import { SpinnerIcon } from './Icons'
import { riskBadgeClass } from './ScoreDisplay'

/**
 * ComparePanel.tsx -- before/after of a proposed design change.
 *
 * The live result is the baseline. Candidate edits stay local until the user
 * runs the comparison, so exploring a fix does not overwrite the assessment.
 */

const PERCENT_KEYS = new Set(['shielding_quality', 'input_filter_quality'])
const CM_CHOKE_MAX_MH = 2

function nudge(
  parameters: DeviceParameters,
  patch: Partial<DeviceParameters>,
): DeviceParameters {
  return { ...parameters, ...patch }
}

export function ComparePanel({
  baseline,
  specs,
  devices,
}: {
  baseline: PredictionResult
  specs: ParameterSpec[]
  devices: DeviceProfile[]
}) {
  const [candidate, setCandidate] = useState<DeviceParameters>(baseline.parameters)
  const [result, setResult] = useState<CompareResult | null>(null)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const dirty = useMemo(
    () => JSON.stringify(candidate) !== JSON.stringify(baseline.parameters),
    [candidate, baseline.parameters],
  )

  const run = async () => {
    setWorking(true)
    setError(null)
    try {
      const comparison = await comparePredictions({
        baseline: baseline.parameters,
        candidate,
        baseline_name: baseline.device_name,
        candidate_name: 'Proposed change',
      })
      setResult(comparison)
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : 'The comparison could not be completed.',
      )
    } finally {
      setWorking(false)
    }
  }

  return (
    <Card className="sm:p-7">
      <Eyebrow>Compare mode</Eyebrow>
      <p className="mt-1 mb-4 text-sm text-ink-muted">
        Hold this assessment as the baseline and try a proposed fix beside it.
        Both configurations are re-simulated.
      </p>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className="btn-secondary text-xs"
          onClick={() =>
            setCandidate(
              nudge(baseline.parameters, {
                cable_length_m: Math.max(1, baseline.parameters.cable_length_m * 0.6),
              }),
            )
          }
        >
          Shorter cable
        </button>
        <button
          type="button"
          className="btn-secondary text-xs"
          onClick={() =>
            setCandidate(
              nudge(baseline.parameters, {
                shielding_quality: Math.min(1, baseline.parameters.shielding_quality + 0.15),
              }),
            )
          }
        >
          Better shield
        </button>
        <button
          type="button"
          className="btn-secondary text-xs"
          onClick={() =>
            setCandidate(nudge(baseline.parameters, { pwm_modulation_type: 'DPWM' }))
          }
        >
          Switch to DPWM
        </button>
        {devices.slice(0, 3).map((device) => (
          <button
            key={device.id}
            type="button"
            className="btn-ghost text-xs"
            onClick={() => setCandidate(device.parameters)}
          >
            vs {device.name}
          </button>
        ))}
      </div>

      <dl className="mt-4 grid gap-2 sm:grid-cols-2">
        {specs.map((spec) => {
          const isPercent = PERCENT_KEYS.has(spec.key)
          const isChoke = spec.key === 'cm_choke_effectiveness'
          const scale = isPercent ? 100 : isChoke ? CM_CHOKE_MAX_MH : 1
          const value = candidate[spec.key] as number
          return (
            <label key={spec.key} className="flex items-center justify-between gap-3 text-xs">
              <span className="text-ink-muted">
                {isChoke ? 'Common-mode choke (mH)' : spec.label}
              </span>
              <input
                type="number"
                className="input-base w-28 py-1 text-right"
                min={spec.min * scale}
                max={spec.max * scale}
                step={isChoke ? 0.1 : spec.step * scale}
                value={isPercent ? Math.round(value * scale) : Number((value * scale).toFixed(2))}
                onChange={(event) => {
                  const next = Number(event.target.value)
                  setCandidate({
                    ...candidate,
                    [spec.key]: next / scale,
                  })
                }}
              />
            </label>
          )
        })}
      </dl>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="btn-primary"
          disabled={!dirty || working}
          onClick={() => void run()}
        >
          {working ? (
            <>
              <SpinnerIcon className="h-4 w-4" />
              Comparing…
            </>
          ) : (
            'Run comparison'
          )}
        </button>
        {dirty ? (
          <button
            type="button"
            className="btn-ghost text-sm"
            onClick={() => {
              setCandidate(baseline.parameters)
              setResult(null)
            }}
          >
            Reset candidate
          </button>
        ) : null}
      </div>

      {error ? <p className="mt-3 text-xs text-fail">{error}</p> : null}

      {result ? <CompareOutcome comparison={result} /> : null}
    </Card>
  )
}

function CompareOutcome({ comparison }: { comparison: CompareResult }) {
  const improved = comparison.delta_risk_score > 0
  return (
    <div className="mt-6 border-t border-line pt-5">
      <div className="grid gap-4 sm:grid-cols-2">
        <ScoreCard title="Baseline" result={comparison.baseline} />
        <ScoreCard title="Proposed change" result={comparison.candidate} />
      </div>
      <p className="mt-4 text-sm text-ink">
        Δ risk score{' '}
        <span className="tabular font-semibold">
          {comparison.delta_risk_score >= 0 ? '+' : ''}
          {comparison.delta_risk_score.toFixed(1)}
        </span>
        <span className="ml-2 text-ink-muted">
          {improved
            ? 'The proposed change increased simulated headroom.'
            : 'The proposed change reduced simulated headroom.'}
        </span>
      </p>
      <table className="mt-3 w-full text-left text-xs">
        <thead>
          <tr className="border-b border-line text-ink-faint">
            <th className="py-2 font-medium">Band</th>
            <th className="py-2 text-right font-medium">Baseline</th>
            <th className="py-2 text-right font-medium">Candidate</th>
            <th className="py-2 text-right font-medium">Δ margin</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {comparison.band_diff.map((row) => (
            <tr key={row.key}>
              <td className="py-2 text-ink">{row.label}</td>
              <td className="py-2 text-right tabular text-ink-muted">
                {row.baseline_margin_db >= 0 ? '+' : ''}
                {row.baseline_margin_db.toFixed(1)} dB
              </td>
              <td className="py-2 text-right tabular text-ink">
                {row.candidate_margin_db >= 0 ? '+' : ''}
                {row.candidate_margin_db.toFixed(1)} dB
              </td>
              <td
                className={`py-2 text-right tabular ${
                  row.delta_margin_db >= 0 ? 'text-pass' : 'text-fail'
                }`}
              >
                {row.delta_margin_db >= 0 ? '+' : ''}
                {row.delta_margin_db.toFixed(1)} dB
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ScoreCard({
  title,
  result,
}: {
  title: string
  result: PredictionResult
}) {
  return (
    <div className="rounded-lg border border-line p-4">
      <p className="text-2xs uppercase tracking-label text-ink-faint">{title}</p>
      <p className="mt-1 text-2xl font-semibold tabular text-ink">
        {result.risk_score.toFixed(0)}
        <span className="ml-1 text-sm font-normal text-ink-faint">
          ±{result.risk_score_plus_minus.toFixed(0)}
        </span>
      </p>
      <span
        className={`mt-2 inline-flex rounded-full border px-2 py-0.5 text-2xs font-semibold uppercase tracking-label ${riskBadgeClass(result.risk_level)}`}
      >
        {result.risk_label}
      </span>
    </div>
  )
}
