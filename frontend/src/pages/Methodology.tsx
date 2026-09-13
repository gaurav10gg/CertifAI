import type { FeatureSpecInfo, MethodologyResponse } from '../api/types'
import { Eyebrow } from '../components/Card'
import { SpinnerIcon } from '../components/Icons'

/**
 * Methodology.tsx -- the "how does this work, and what should I not trust"
 * document, rendered inside the methodology sheet.
 *
 * Content comes from GET /api/methodology rather than being written into the
 * frontend, so the numbers quoted here are the ones the deployed model actually
 * reports. The two sections that matter most are the limit-curve assumption and
 * the design-only ablation -- both are places where a less honest tool would
 * quote a flattering number instead.
 */

export function Methodology({
  data,
  loading,
  error,
}: {
  data: MethodologyResponse | null
  loading: boolean
  error: string | null
}) {
  if (loading) {
    return (
      <div className="flex items-center gap-2.5 text-sm text-ink-muted">
        <SpinnerIcon className="h-4 w-4" />
        Loading methodology…
      </div>
    )
  }

  if (error || !data) {
    return (
      <p className="text-sm text-fail">
        {error ?? 'The methodology document could not be loaded.'}
      </p>
    )
  }

  const validation = data.validation
  const ablation = validation.design_only_ablation

  return (
    <div className="space-y-9">
      <Section
        step="01"
        title="Physics simulation"
        lead="Every assessment starts from a synthesised switching waveform, not from a lookup table."
      >
        <p>
          The inverter's three legs are switched with the selected PWM strategy at
          the chosen carrier frequency. From the three leg voltages the model forms
          the common-mode voltage <Mono>(v_a + v_b + v_c) / 3</Mono> — a staircase
          that steps six times per carrier period and is the quantity that actually
          drives conducted emissions on the motor cable.
        </p>
        <p>
          Edges are trapezoidal with a rise time set by <Mono>dv/dt</Mono>, which
          fixes where the spectral envelope starts rolling off. The common-mode
          displacement current through the cable is{' '}
          <Mono>i = C · dv/dt</Mono> with 100 pF of parasitic capacitance per metre,
          then shaped by the cable's transmission-line resonances (a quarter-wave
          mode plus its third and fifth harmonics), attenuated by the shield, scaled
          by load current, and finally summed with a randomised noise floor.
        </p>
        <p>
          Because a single short capture would only see one slice of the 50 Hz output
          cycle, the simulator takes{' '}
          <Mono>{data.simulation.n_dwell_segments}</Mono> short records spread across
          one fundamental period and max-holds them. That is what a real EMI receiver
          does when it dwells on a frequency, and it removes the duty-cycle sampling
          variance that would otherwise swamp the parameter effects.
        </p>
        <KeyValues
          items={[
            ['Sample rate', `${(data.simulation.sample_rate_hz / 1e6).toFixed(0)} MHz`],
            ['Dwell segments', String(data.simulation.n_dwell_segments)],
            [
              'Resolution bandwidth',
              `${(data.simulation.receiver_rbw_hz / 1e3).toFixed(0)} kHz (CISPR 16-1-1 band B)`,
            ],
          ]}
        />
      </Section>

      <Section
        step="02"
        title="Spectral analysis"
        lead="The waveform is reduced to 18 numbers via a receiver-emulated FFT."
      >
        <p>
          Each segment is windowed and transformed, the segments are max-held, and
          power is integrated across a {(data.simulation.receiver_rbw_hz / 1e3).toFixed(0)}{' '}
          kHz resolution bandwidth to give the trace a test house would actually
          read. Within each band the model extracts four features: peak amplitude,
          RMS amplitude, the number of spectral lines within{' '}
          {data.simulation.harmonic_proximity_db.toFixed(0)} dB of the limit, and a
          THD-like noisiness score.
        </p>
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-line">
              <th className="py-2 pr-3 font-medium text-ink-faint">Band</th>
              <th className="py-2 pr-3 font-medium text-ink-faint">Range</th>
              <th className="py-2 font-medium text-ink-faint">Assumed limit</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {data.bands.map((band) => (
              <tr key={band.key}>
                <td className="py-2 pr-3 text-ink">{band.key.replace('band_', 'Band ').toUpperCase()}</td>
                <td className="py-2 pr-3 tabular text-ink-muted">{band.label}</td>
                <td className="py-2 tabular text-ink-muted">
                  {band.limit_low_dbuv.toFixed(0)} → {band.limit_high_dbuv.toFixed(0)}{' '}
                  dBµV
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      {/* The single most important caveat in the product. Given its own visual
          weight rather than buried in a paragraph. */}
      <section className="rounded-card border border-line border-l-2 border-l-ink bg-surface p-5">
        <Eyebrow>The limit curve is an assumption</Eyebrow>
        <h3 className="mt-2 text-base font-semibold tracking-tight text-ink">
          Read this before trusting any number here
        </h3>
        <div className="mt-2.5 space-y-3 text-sm leading-relaxed text-ink-muted">
          <p>{data.limit_curve.description}</p>
          <p>{data.limit_curve.provenance}</p>
          <p>
            The consequence is specific: an absolute pass or fail from this tool is
            only as correct as that assumed curve. What the tool does reliably tell
            you is the <em className="not-italic text-ink">direction and size</em> of
            a change — how much a shorter cable, a better shield or a slower{' '}
            <Mono>dv/dt</Mono> moves the emission level. That is what makes it useful
            during early design, and it is why the limit is stated as an anchor list
            you can replace.
          </p>
        </div>
        <dl className="mt-4 flex flex-wrap gap-x-6 gap-y-2 border-t border-line pt-3.5">
          {data.limit_curve.anchors_hz_dbuv.map(([hz, dbuv]) => (
            <div key={hz} className="flex items-baseline gap-2">
              <dt className="text-2xs uppercase tracking-label text-ink-faint">
                {hz >= 1e6 ? `${hz / 1e6} MHz` : `${hz / 1e3} kHz`}
              </dt>
              <dd className="text-sm tabular text-ink">{dbuv.toFixed(0)} dBµV</dd>
            </div>
          ))}
        </dl>
      </section>

      <Section
        step="03"
        title="Physics-informed machine learning"
        lead="Monotonic constraints are what make this more than curve fitting."
      >
        <p>
          Two gradient-boosted models are trained per band on{' '}
          {validation.n_samples?.toLocaleString() ?? 'several thousand'} simulated
          designs: a classifier for pass/fail and a regressor for the continuous
          margin in dB. Both are trained with XGBoost's{' '}
          <Mono>monotone_constraints</Mono>.
        </p>
        <p>
          In plain language: the model is <em className="not-italic text-ink">forbidden</em>{' '}
          from learning a relationship that contradicts physics. It cannot decide
          that a longer cable reduces emissions, or that a better shield makes things
          worse, even if a pocket of the training data happens to suggest it. Every
          split on every tree is required to move risk in the physically correct
          direction. The constraint is enforced structurally at training time, not
          checked afterwards.
        </p>
        <p className="text-ink">{data.monotone_constraints.explanation}</p>

        <FeatureTable
          caption="Design parameters and their constrained direction"
          features={data.features.design}
        />

        {validation.monotonicity_audit ? (
          <p className="text-xs text-ink-faint">
            Verified post-training by sweeping each parameter across its range:{' '}
            {validation.monotonicity_audit.filter((row) => row.respects_constraint).length}{' '}
            of {validation.monotonicity_audit.length} constraints hold in the trained
            models.
          </p>
        ) : null}
      </Section>

      <Section
        step="04"
        title="What the accuracy figures mean"
        lead="These are simulation-consistency scores. They are not real-world accuracy."
      >
        <p>{validation.metric_semantics}</p>

        {validation.band_metrics ? (
          <MetricsTable
            caption="Held-out agreement with the simulator, all 18 features"
            rows={validation.band_metrics.map((row) => ({
              band: row.band_label,
              accuracy: row.classifier_balanced_accuracy,
              mae: row.margin_mae_db,
            }))}
          />
        ) : null}

        {ablation ? (
          <>
            <p>{ablation.note}</p>
            <MetricsTable
              caption="Design parameters only — the figure that reflects genuine predictive work"
              rows={ablation.band_metrics.map((row) => ({
                band: row.band_label,
                accuracy: row.classifier_balanced_accuracy,
                mae: row.margin_mae_db,
              }))}
            />
          </>
        ) : null}

        {validation.seed_stability ? (
          <p className="text-xs leading-relaxed text-ink-faint">
            Repeat-noise check: re-simulating an identical design with different
            noise seeds moves the measured margin by{' '}
            {validation.seed_stability.simulator_margin_sd_db.toFixed(2)} dB
            (standard deviation), and the model's prediction by{' '}
            {validation.seed_stability.model_margin_sd_db.toFixed(2)} dB. The model
            therefore does not smooth away simulator noise — it tracks it. Reported
            here because it is a measured negative result, not a selling point.
          </p>
        ) : null}
      </Section>

      <Section
        step="05"
        title="Score, confidence and risk attribution"
        lead="How the three headline numbers are produced."
      >
        <p>
          The <strong className="font-medium text-ink">compliance score</strong> maps
          each band's predicted margin through a saturating function — 0 dB of margin
          becomes 50 — then blends the mean of the three bands with the worst of them
          in equal parts. A design cannot earn a good score by passing two bands
          comfortably while failing the third.
        </p>
        <p>
          The <strong className="font-medium text-ink">confidence score</strong> is
          the probability, under the model's own measured margin error, that each
          band's verdict would survive that error — multiplied across bands, reduced
          when the classifier and the regressor disagree, and capped below 100. The
          cap is deliberate: the dominant error term is the gap between this
          simulation and physical reality, and that term is not quantified anywhere in
          this tool.
        </p>
        <p>
          The <strong className="font-medium text-ink">top risk factor</strong> comes
          from exact tree SHAP values on the worst band's classifier, weighted by how
          far each parameter already sits towards its own risky extreme. A parameter
          already at its safest setting is not proposed for further change.
        </p>
      </Section>

      <Section
        step="06"
        title="Extending this to real measurements"
        lead="The intended path from prototype to instrument."
      >
        <p>
          The backend contains a documented extension point,{' '}
          <Mono>calibrate.py</Mono>, for fitting a residual correction from real
          chamber measurements: measured minus predicted margin, learned as a
          function of the design parameters, then added to the simulator's output.
          That structure means real data improves the tool without discarding the
          physics or the monotonicity guarantees. It is intentionally left
          unimplemented rather than faked.
        </p>
      </Section>

      <p className="border-t border-line pt-5 text-xs leading-relaxed text-ink-faint">
        {data.disclaimer_long}
      </p>
    </div>
  )
}

function Section({
  step,
  title,
  lead,
  children,
}: {
  step: string
  title: string
  lead: string
  children: React.ReactNode
}) {
  return (
    <section>
      <div className="flex items-baseline gap-3">
        <span className="text-xs font-semibold tabular text-ink-faint">{step}</span>
        <h3 className="text-base font-semibold tracking-tight text-ink">{title}</h3>
      </div>
      <p className="mt-1.5 pl-8 text-sm font-medium text-ink">{lead}</p>
      <div className="mt-3 space-y-3 pl-8 text-sm leading-relaxed text-ink-muted">
        {children}
      </div>
    </section>
  )
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded border border-line bg-surface px-1 py-0.5 text-2xs tabular text-ink">
      {children}
    </code>
  )
}

function KeyValues({ items }: { items: [string, string][] }) {
  return (
    <dl className="grid gap-x-6 gap-y-2 border-t border-line pt-3 sm:grid-cols-3">
      {items.map(([label, value]) => (
        <div key={label}>
          <dt className="text-2xs uppercase tracking-label text-ink-faint">
            {label}
          </dt>
          <dd className="mt-0.5 text-xs tabular text-ink">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

function FeatureTable({
  caption,
  features,
}: {
  caption: string
  features: FeatureSpecInfo[]
}) {
  return (
    <div>
      <p className="label-eyebrow">{caption}</p>
      <table className="mt-2 w-full text-left text-xs">
        <tbody className="divide-y divide-line">
          {features.map((feature) => (
            <tr key={feature.name}>
              <td className="w-6 py-2 pr-2 align-top text-sm font-semibold tabular text-ink">
                {feature.risk_sign > 0 ? '↑' : '↓'}
              </td>
              <td className="py-2 align-top leading-relaxed text-ink-muted">
                {feature.rationale}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1.5 text-2xs text-ink-faint">
        ↑ increasing this parameter may only increase predicted risk. ↓ increasing it
        may only decrease it.
      </p>
    </div>
  )
}

function MetricsTable({
  caption,
  rows,
}: {
  caption: string
  rows: { band: string; accuracy: number; mae: number }[]
}) {
  return (
    <div>
      <p className="label-eyebrow">{caption}</p>
      <table className="mt-2 w-full text-left text-xs">
        <thead>
          <tr className="border-b border-line">
            <th className="py-2 pr-3 font-medium text-ink-faint">Band</th>
            <th className="py-2 pr-3 text-right font-medium text-ink-faint">
              Balanced accuracy
            </th>
            <th className="py-2 text-right font-medium text-ink-faint">
              Margin MAE
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((row) => (
            <tr key={row.band}>
              <td className="py-2 pr-3 tabular text-ink">{row.band}</td>
              <td className="py-2 pr-3 text-right tabular text-ink-muted">
                {row.accuracy.toFixed(3)}
              </td>
              <td className="py-2 text-right tabular text-ink-muted">
                {row.mae.toFixed(2)} dB
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
