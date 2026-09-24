import type {
  AssessmentHistoryEntry,
  DeviceProfile,
  ParameterSpec,
  PredictionResult,
  RadiatedAssessment,
  TradeoffResponse,
} from '../api/types'
import { Card, Eyebrow } from '../components/Card'
import { BandBreakdown } from '../components/BandBreakdown'
import { CertificateButton } from '../components/CertificateButton'
import { ComparePanel } from '../components/ComparePanel'
import { CountermeasureList } from '../components/CountermeasureList'
import { Disclaimer } from '../components/Disclaimer'
import { HistorySparkline } from '../components/HistorySparkline'
import { ArrowLeftIcon } from '../components/Icons'
import { ScoreDisplay } from '../components/ScoreDisplay'
import { ShapWaterfall } from '../components/ShapWaterfall'
import { SignalExplorer } from '../components/SignalExplorer'
import { SpectrumChart } from '../components/SpectrumChart'
import { ThdPanel } from '../components/ThdPanel'
import { TornadoChart } from '../components/TornadoChart'
import { TradeoffExplorer } from '../components/TradeoffExplorer'

/**
 * Results.tsx -- stage three: the risk assessment.
 *
 * Framing sentence first, then the three-tier risk badge and score with
 * uncertainty. Per-band margins stay visible underneath: the tier is a summary,
 * not a replacement for the detail.
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
  history,
  specs,
  devices,
  tradeoff,
  tradeoffLoading,
  tradeoffError,
  applyingCarrier,
  applyingFrequency,
  applyError,
  onApplyCarrier,
  onBack,
  onRestart,
  onOpenMethodology,
  onOpenValidation,
}: {
  result: PredictionResult
  history: AssessmentHistoryEntry[]
  specs: ParameterSpec[]
  devices: DeviceProfile[]
  tradeoff: TradeoffResponse | null
  tradeoffLoading: boolean
  tradeoffError: string | null
  applyingCarrier: boolean
  applyingFrequency: number | null
  applyError: string | null
  onApplyCarrier: (khz: number) => void
  onBack: () => void
  onRestart: () => void
  onOpenMethodology: () => void
  onOpenValidation: () => void
}) {
  const consistency = result.model_info.simulation_consistency
  const ablation = consistency.design_only_ablation
  const framing =
    result.framing ||
    'This tool estimates EMC risk from simulated physics. It does not predict EN 12016 certification outcomes, which require accredited lab measurement.'

  return (
    <div className="space-y-6 motion-safe:animate-fade-up">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Eyebrow>Step 3 · Pre-compliance risk indicator</Eyebrow>
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

      <p className="rounded-card border border-line border-l-2 border-l-ink bg-surface px-5 py-3.5 text-sm leading-relaxed text-ink">
        {framing}
      </p>

      {history.length > 1 ? <HistorySparkline history={history} /> : null}

      <Card className="sm:p-8">
        <ScoreDisplay
          score={result.risk_score}
          plusMinus={result.risk_score_plus_minus}
          scoreLow={result.risk_score_low}
          scoreHigh={result.risk_score_high}
          riskLevel={result.risk_level}
          riskLabel={result.risk_label}
          riskCopy={result.risk_copy}
          confidence={result.confidence_score}
          confidenceLabel={result.confidence_label}
          confidenceNote={result.confidence_note}
        />
      </Card>

      <Card className="sm:p-7">
        <SpectrumChart spectrum={result.spectrum} bands={result.bands} />
      </Card>

      <Card className="sm:p-7">
        <Eyebrow>Band margins</Eyebrow>
        <p className="mt-1 mb-4 text-sm text-ink-muted">
          Headroom below the assumed limit at each band's worst frequency.
          Negative means the simulated emission exceeds that assumed curve. This
          table is the detail; the risk tier above is only a summary.
        </p>
        <BandBreakdown bands={result.bands} />
      </Card>

      {result.radiated ? <RadiatedSection radiated={result.radiated} /> : null}

      {result.signals?.length ? (
        <Card className="sm:p-7">
          <SignalExplorer signals={result.signals} />
        </Card>
      ) : null}

      {result.power_quality?.length ? (
        <Card className="sm:p-7">
          <ThdPanel reports={result.power_quality} />
        </Card>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-2 lg:items-start">
        {result.shap?.contributions?.length ? (
          <Card className="sm:p-7">
            <ShapWaterfall
              contributions={result.shap.contributions}
              band={result.shap.band}
            />
          </Card>
        ) : null}
        {result.sensitivity?.bars?.length ? (
          <Card className="sm:p-7">
            <TornadoChart
              bars={result.sensitivity.bars}
              note={result.sensitivity.note}
            />
          </Card>
        ) : null}
      </div>

      {result.shap?.contributions?.length && result.countermeasures?.length ? (
        <p className="text-sm leading-relaxed text-ink-muted">
          These lists can diverge: the chart explains why the current design sits
          where it does, including factors already optimized (like shielding). The
          recommendations below only suggest parameters with room left to improve.
        </p>
      ) : null}

      {result.countermeasures?.length ? (
        <Card className="sm:p-7">
          <CountermeasureList measures={result.countermeasures} />
        </Card>
      ) : null}

      <TradeoffExplorer
        data={tradeoff}
        loading={tradeoffLoading}
        error={tradeoffError ?? applyError}
        currentFrequency={result.parameters.switching_frequency_khz}
        applying={applyingCarrier}
        applyingFrequency={applyingFrequency}
        onSelectFrequency={onApplyCarrier}
      />

      <ComparePanel baseline={result} specs={specs} devices={devices} />

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
              hint={`all ${result.model_info.feature_count} features`}
            />
            <ConsistencyRow
              label="Margin error"
              value={formatDb(consistency.mean_margin_mae_db)}
              hint="mean absolute, all features"
            />
            <ConsistencyRow
              label="Design-only accuracy"
              value={formatRatio(ablation.mean_balanced_accuracy)}
              hint={`${result.model_info.design_feature_count ?? 7} design parameters only`}
            />
            <ConsistencyRow
              label="Design-only margin error"
              value={formatDb(ablation.mean_margin_mae_db)}
              hint="the honest figure"
            />
          </dl>

          <p className="mt-3.5 text-2xs leading-relaxed text-ink-faint">
            {ablation.note}{' '}
            <button
              type="button"
              onClick={onOpenValidation}
              className="rounded underline decoration-line-strong underline-offset-2
                transition-colors hover:text-ink-muted hover:decoration-ink-muted"
            >
              Model validation
            </button>
            : monotonicity and SHAP identity, with the measured numbers.
          </p>
        </Card>
      </div>

      <Card className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between sm:p-7">
        <div className="max-w-xl">
          <Eyebrow>Virtual EMC Pre-Compliance Report</Eyebrow>
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">
            A PDF of the risk level, uncertainty band, spectrum, Why this
            margin, what to change next, the design trade-off context, five
            signal traces, and THD. Recomputed server-side from the
            parameters, so the document always matches the model.
          </p>
        </div>
        <CertificateButton result={result} />
      </Card>

      <Card>
        <Eyebrow>Assumptions and limitations</Eyebrow>
        <p className="mt-2 text-sm leading-relaxed text-ink-muted">
          {result.assumptions_scope}
        </p>
      </Card>

      <Disclaimer
        text={result.disclaimer}
        onOpenMethodology={onOpenMethodology}
        className="border-t border-line pt-5"
      />
    </div>
  )
}

function RadiatedSection({ radiated }: { radiated: RadiatedAssessment }) {
  return (
    <section className="rounded-card border border-dashed border-line-strong bg-surface/70 p-5 sm:p-7">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Eyebrow>Separate assessment · 30 MHz–1 GHz</Eyebrow>
          <h2 className="mt-1 text-lg font-semibold tracking-tight text-ink">
            {radiated.title}
          </h2>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-2xl font-semibold tabular leading-none text-ink">
            {Math.round(radiated.risk_score)}
            <span className="ml-1 text-sm font-normal text-ink-faint">/100</span>
          </span>
          <span
            className="inline-flex items-center rounded-full border border-dashed
              border-ink-faint px-2.5 py-1 text-2xs font-semibold uppercase
              tracking-label text-ink-muted"
          >
            {radiated.badge}
          </span>
        </div>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-ink-muted">{radiated.caption}</p>
      <p className="mt-2 text-sm text-ink">
        <span className="text-ink-faint">Top driver · </span>
        {radiated.top_factor.statement}
      </p>

      <div className="mt-6">
        <SpectrumChart
          spectrum={radiated.spectrum}
          bands={radiated.bands}
          eyebrow="Radiated spectrum"
          subtitle="Clock harmonics and the synthetic radiated limit, 30 MHz to 1 GHz."
          unit="dBµV/m"
          yLabel="dBµV/m"
          xTicks={[30, 50, 100, 230, 500, 1000]}
          dividersMhz={[230]}
          chartId="radiated"
          showDots
        />
      </div>

      <div className="mt-6">
        <Eyebrow>Radiated bands</Eyebrow>
        <div className="mt-3">
          <BandBreakdown bands={radiated.bands} unit="dBµV/m" showUncertainty={false} />
        </div>
      </div>

      <p className="mt-5 border-t border-dashed border-line pt-4 text-xs leading-relaxed text-ink-muted">
        {radiated.disclaimer}
      </p>
    </section>
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
