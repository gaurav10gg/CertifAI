import type { ReactNode } from 'react'

import type { ValidationResponse } from '../api/types'
import { Eyebrow } from '../components/Card'
import { SpinnerIcon } from '../components/Icons'

/**
 * ModelValidation.tsx -- measured checks, not claims.
 *
 * The two sentences a judge should be able to read in five seconds live at
 * the top. Everything under them is the arithmetic that produced those
 * sentences, served from GET /api/validation.
 */

export function ModelValidation({
  data,
  loading,
  error,
}: {
  data: ValidationResponse | null
  loading: boolean
  error: string | null
}) {
  if (loading) {
    return (
      <div className="flex items-center gap-2.5 text-sm text-ink-muted">
        <SpinnerIcon className="h-4 w-4" />
        Loading validation suite…
      </div>
    )
  }

  if (error || !data) {
    return (
      <p className="text-sm text-fail">
        {error ?? 'The validation suite could not be loaded.'}
      </p>
    )
  }

  const features = Object.values(data.monotonicity.per_feature)
  const ablation = data.simulation_consistency.design_only_ablation?.band_metrics

  return (
    <div className="space-y-8">
      <p className="text-sm leading-relaxed text-ink-muted">
        {data.note}
      </p>

      <CheckCard
        passed={data.monotonicity.passed}
        eyebrow="Monotonicity"
        headline={data.headlines.monotonicity}
        meta={`${data.monotonicity.n_configs} configs · ${data.monotonicity.n_checks} sweeps · ${data.monotonicity.n_violations} violations`}
        note={data.monotonicity.note}
      >
        <table className="mt-3 w-full text-left text-xs">
          <thead>
            <tr className="border-b border-line">
              <th className="py-2 pr-3 font-medium text-ink-faint">Parameter</th>
              <th className="py-2 pr-3 text-right font-medium text-ink-faint">
                Sweeps
              </th>
              <th className="py-2 text-right font-medium text-ink-faint">
                Violations
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {features.map((row) => (
              <tr key={row.label}>
                <td className="py-2 pr-3 text-ink">
                  {row.label}
                  <span className="ml-2 tabular text-ink-faint">
                    {row.risk_sign > 0 ? '↑ risk' : '↓ risk'}
                  </span>
                </td>
                <td className="py-2 pr-3 text-right tabular text-ink-muted">
                  {row.checked}
                </td>
                <td className="py-2 text-right tabular text-ink">
                  {row.violations}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CheckCard>

      <CheckCard
        passed={data.shap_additivity.passed}
        eyebrow="SHAP additivity"
        headline={data.headlines.shap_additivity}
        meta={`${data.shap_additivity.n_checks} identities · max |error| ${data.shap_additivity.max_abs_error_display} dB`}
        note={data.shap_additivity.note}
      >
        <table className="mt-3 w-full text-left text-xs">
          <thead>
            <tr className="border-b border-line">
              <th className="py-2 pr-3 font-medium text-ink-faint">Band</th>
              <th className="py-2 pr-3 text-right font-medium text-ink-faint">
                Mean |error|
              </th>
              <th className="py-2 text-right font-medium text-ink-faint">
                Max |error|
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {data.shap_additivity.per_band.map((row) => (
              <tr key={row.band}>
                <td className="py-2 pr-3 tabular text-ink">{row.band}</td>
                <td className="py-2 pr-3 text-right tabular text-ink-muted">
                  {formatSci(row.mean_abs_error_db)} dB
                </td>
                <td className="py-2 text-right tabular text-ink-muted">
                  {formatSci(row.max_abs_error_db)} dB
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CheckCard>

      {ablation?.length ? (
        <section>
          <Eyebrow>Held-out design-only agreement</Eyebrow>
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">
            Same models, spectrum hidden: six design parameters only,{' '}
            {data.simulation_consistency.n_test?.toLocaleString() ?? 'held-out'}{' '}
            test designs. This is simulation-consistency, not lab accuracy.
          </p>
          <table className="mt-3 w-full text-left text-xs">
            <thead>
              <tr className="border-b border-line">
                <th className="py-2 pr-3 font-medium text-ink-faint">Band</th>
                <th className="py-2 pr-3 text-right font-medium text-ink-faint">
                  Balanced acc.
                </th>
                <th className="py-2 text-right font-medium text-ink-faint">
                  Margin MAE
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {ablation.map((row) => (
                <tr key={row.band_label}>
                  <td className="py-2 pr-3 tabular text-ink">{row.band_label}</td>
                  <td className="py-2 pr-3 text-right tabular text-ink-muted">
                    {row.classifier_balanced_accuracy.toFixed(3)}
                  </td>
                  <td className="py-2 text-right tabular text-ink-muted">
                    {row.margin_mae_db.toFixed(2)} dB
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}

      <p className="border-t border-line pt-4 text-2xs leading-relaxed text-ink-faint">
        Suite {data.artifact_version}
        {data.generated_at ? ` · ran ${data.generated_at.replace('T', ' ')}` : ''}
        . Regenerated with{' '}
        <code className="rounded border border-line bg-surface px-1 py-0.5">
          python tools/run_validation.py
        </code>
        .
      </p>
    </div>
  )
}

function CheckCard({
  passed,
  eyebrow,
  headline,
  meta,
  note,
  children,
}: {
  passed: boolean
  eyebrow: string
  headline: string
  meta: string
  note: string
  children: ReactNode
}) {
  return (
    <section className="rounded-card border border-line border-l-2 bg-surface p-5 border-l-ink">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <Eyebrow>{eyebrow}</Eyebrow>
        <span
          className={`text-2xs font-semibold uppercase tracking-label ${
            passed ? 'text-pass' : 'text-fail'
          }`}
        >
          {passed ? 'Verified' : 'Failed'}
        </span>
      </div>
      <p className="mt-2 text-base font-semibold tracking-tight text-ink">
        {headline}
      </p>
      <p className="mt-1 text-xs tabular text-ink-faint">{meta}</p>
      {children}
      <p className="mt-3 text-2xs leading-relaxed text-ink-faint">{note}</p>
    </section>
  )
}

function formatSci(value: number): string {
  return value
    .toExponential(1)
    .replace(/e([+-])0(\d)/, 'e$1$2')
}
