import type { PredictionResult } from '../api/types'
import { Card, Eyebrow } from '../components/Card'
import { BandBreakdown } from '../components/BandBreakdown'
import { CertificateButton } from '../components/CertificateButton'
import { Disclaimer } from '../components/Disclaimer'
import { ArrowLeftIcon } from '../components/Icons'
import { RiskCallout } from '../components/RiskCallout'
import { ScoreDisplay } from '../components/ScoreDisplay'
import { SpectrumChart } from '../components/SpectrumChart'

/**
 * Results.tsx -- stage three: the assessment.
 *
 * Reading order is deliberate. Score and verdict first, because that is the
 * question the user asked. Then the spectrum, which is the evidence. Then the
 * per-band numbers. Then what to do about it. The disclaimer sits at the foot of
 * the page as a standing footnote and is repeated in the PDF.
 */

function formatTimestamp(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

export function Results({
  result,
  onBack,
  onRestart,
  onOpenMethodology,
}: {
  result: PredictionResult
  onBack: () => void
  onRestart: () => void
  onOpenMethodology: () => void
}) {
  const consistency = result.model_info.simulation_consistency
  const ablation = consistency.design_only_ablation

  return (
    <div className="space-y-6 motion-safe:animate-fade-up">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Eyebrow>Step 3 · Virtual pre-compliance assessment</Eyebrow>
          <h1 className="mt-2 text-display font-semibold tracking-tight text-ink">
            {result.device_name}
          </h1>
          <p className="mt-1.5 text-sm tabular text-ink-faint">
            Assessed {formatTimestamp(result.generated_at)} · EN 12016-style
            conducted emissions
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={onBack} className="btn-secondary">
            <ArrowLeftIcon className="h-4 w-4" />
            Adjust parameters
          </button>
          <button type="button" onClick={onRestart} className="btn-ghost text-sm">
            New assessment
          </button>
        </div>
      </header>

      <Card className="sm:p-8">
        <ScoreDisplay
          score={result.compliance_score}
          confidence={result.confidence_score}
          confidenceLabel={result.confidence_label}
          confidenceNote={result.confidence_note}
          verdict={result.verdict}
        />
      </Card>

      <Card className="sm:p-7">
        <SpectrumChart spectrum={result.spectrum} bands={result.bands} />
      </Card>

      <Card className="sm:p-7">
        <Eyebrow>Band breakdown</Eyebrow>
        <p className="mt-1 mb-4 text-sm text-ink-muted">
          Margin is headroom below the assumed limit at the band's worst
          frequency. Negative means the band is predicted to fail.
        </p>
        <BandBreakdown bands={result.bands} />
      </Card>

      <RiskCallout risk={result.top_risk_factor} />

      <div className="grid gap-6 lg:grid-cols-[1fr_1fr] lg:items-start">
        <Card>
          <Eyebrow>Configuration assessed</Eyebrow>
          <dl className="mt-3 divide-y divide-line">
            {result.parameter_display.map((row) => (
              <div
                key={row.key}
                className="flex items-baseline justify-between gap-4 py-2"
              >
                <dt className="text-sm text-ink-muted">{row.label}</dt>
                <dd className="text-sm tabular text-ink">{row.formatted}</dd>
              </div>
            ))}
          </dl>
        </Card>

        <Card>
          <Eyebrow>Simulation-consistency</Eyebrow>
          <p className="mt-2 text-xs leading-relaxed text-ink-muted">
            {consistency.note}
          </p>

          <dl className="mt-4 space-y-2.5 border-t border-line pt-3.5">
            <ConsistencyRow
              label="Balanced accuracy"
              value={formatRatio(consistency.mean_balanced_accuracy)}
              hint="all 18 features"
            />
            <ConsistencyRow
              label="Margin error"
              value={formatDb(consistency.mean_margin_mae_db)}
              hint="mean absolute, all features"
            />
            <ConsistencyRow
              label="Design-only accuracy"
              value={formatRatio(ablation.mean_balanced_accuracy)}
              hint="6 design parameters only"
            />
            <ConsistencyRow
              label="Design-only margin error"
              value={formatDb(ablation.mean_margin_mae_db)}
              hint="the honest figure"
            />
          </dl>

          <p className="mt-3.5 text-2xs leading-relaxed text-ink-faint">
            {ablation.note}
          </p>
        </Card>
      </div>

      <Card className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between sm:p-7">
        <div className="max-w-xl">
          <Eyebrow>Assessment document</Eyebrow>
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">
            A two-page PDF containing the configuration, the score, the per-band
            table, the spectrum plot and the stated assumptions. Recomputed
            server-side from the parameters, so the document always matches the
            model.
          </p>
        </div>
        <CertificateButton result={result} />
      </Card>

      <Disclaimer
        text={result.disclaimer}
        onOpenMethodology={onOpenMethodology}
        className="border-t border-line pt-5"
      />
    </div>
  )
}

function ConsistencyRow({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint: string
}) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="text-xs text-ink-muted">
        {label}
        <span className="ml-1.5 text-2xs text-ink-faint">{hint}</span>
      </dt>
      <dd className="shrink-0 text-sm tabular text-ink">{value}</dd>
    </div>
  )
}

function formatRatio(value: number | null): string {
  return value === null ? '—' : value.toFixed(3)
}

function formatDb(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(2)} dB`
}
