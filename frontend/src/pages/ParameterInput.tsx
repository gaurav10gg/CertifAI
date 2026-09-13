import type {
  DeviceParameters,
  ParameterSpec,
  PwmModulationType,
  PwmOption,
} from '../api/types'
import { Card, Eyebrow } from '../components/Card'
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  SpinnerIcon,
} from '../components/Icons'
import { SegmentedControl, Slider } from '../components/Slider'

/**
 * ParameterInput.tsx -- stage two: the six design parameters.
 *
 * Bounds, steps and tooltip text all come from the backend
 * (PARAMETER_RANGES in simulate.py), so the UI cannot drift out of the range the
 * model was trained on. Every control carries a tooltip explaining what the
 * quantity physically represents and which direction increases emissions.
 */

/** Percentage-style parameters are stored 0-1 but are far easier to read as %. */
const PERCENT_KEYS = new Set(['shielding_quality'])

export function ParameterInput({
  specs,
  pwmOptions,
  parameters,
  deviceName,
  isCustom,
  onChange,
  onBack,
  onSubmit,
  onResetToPreset,
  dirty,
  submitting,
  error,
}: {
  specs: ParameterSpec[]
  pwmOptions: PwmOption[]
  parameters: DeviceParameters
  deviceName: string
  isCustom: boolean
  onChange: (next: DeviceParameters) => void
  onBack: () => void
  onSubmit: () => void
  onResetToPreset: () => void
  dirty: boolean
  submitting: boolean
  error: string | null
}) {
  const setNumeric = (key: ParameterSpec['key'], value: number) =>
    onChange({ ...parameters, [key]: value })

  return (
    <div className="motion-safe:animate-fade-up">
      <header className="max-w-2xl">
        <Eyebrow>Step 2</Eyebrow>
        <h1 className="mt-2 text-display font-semibold tracking-tight text-ink">
          {isCustom ? 'Custom configuration' : deviceName}
        </h1>
        <p className="mt-3 text-base leading-relaxed text-ink-muted">
          These six parameters drive the physics simulation. Hover any label to see
          what the quantity represents and how it affects conducted emissions.
        </p>
      </header>

      <div className="mt-9 grid gap-6 lg:grid-cols-[1.55fr_1fr] lg:items-start">
        <Card className="space-y-7">
          {specs.map((spec) => {
            const isPercent = PERCENT_KEYS.has(spec.key)
            return (
              <Slider
                key={spec.key}
                label={isPercent ? `${spec.label} (%)` : spec.label}
                unit={isPercent ? '%' : spec.unit}
                min={isPercent ? spec.min * 100 : spec.min}
                max={isPercent ? spec.max * 100 : spec.max}
                step={isPercent ? spec.step * 100 : spec.step}
                value={
                  isPercent
                    ? parameters[spec.key] * 100
                    : (parameters[spec.key] as number)
                }
                precision={isPercent ? 0 : undefined}
                description={spec.description}
                onChange={(value) =>
                  setNumeric(spec.key, isPercent ? value / 100 : value)
                }
              />
            )
          })}

          <div className="border-t border-line pt-7">
            <SegmentedControl<PwmModulationType>
              label="PWM modulation strategy"
              description="How the three inverter legs are switched. The modelled figure is the
                common-mode emission penalty relative to plain sinusoidal PWM: discontinuous
                PWM commutates roughly a third less often and so emits less, while
                space-vector and randomised PWM inject additional zero-sequence content."
              options={pwmOptions.map((option) => ({
                value: option.value,
                label: option.label,
                hint: `${option.cm_penalty_db >= 0 ? '+' : ''}${option.cm_penalty_db.toFixed(1)} dB CM`,
              }))}
              value={parameters.pwm_modulation_type}
              onChange={(value) =>
                onChange({ ...parameters, pwm_modulation_type: value })
              }
            />
          </div>
        </Card>

        <div className="space-y-4 lg:sticky lg:top-20">
          <Card>
            <Eyebrow>Run the assessment</Eyebrow>
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">
              The backend synthesises the switching waveform, applies a
              receiver-emulated FFT across the three EMC bands, and evaluates the
              monotonically constrained models. This takes about a second.
            </p>

            <button
              type="button"
              onClick={onSubmit}
              disabled={submitting}
              className="btn-primary mt-5 w-full"
            >
              {submitting ? (
                <>
                  <SpinnerIcon className="h-4 w-4" />
                  Running simulation…
                </>
              ) : (
                <>
                  Run compliance check
                  <ArrowRightIcon className="h-4 w-4" />
                </>
              )}
            </button>

            {error ? (
              <p className="mt-3 text-xs leading-relaxed text-fail">{error}</p>
            ) : null}

            <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-line pt-4">
              <button type="button" onClick={onBack} className="btn-ghost text-sm">
                <ArrowLeftIcon className="h-3.5 w-3.5" />
                Change device
              </button>
              {!isCustom && dirty ? (
                <button
                  type="button"
                  onClick={onResetToPreset}
                  className="btn-ghost text-sm"
                >
                  Reset to preset
                </button>
              ) : null}
            </div>
          </Card>

          <Card className="bg-surface">
            <Eyebrow>What is being assessed</Eyebrow>
            <ul className="mt-3 space-y-2.5 text-xs leading-relaxed text-ink-muted">
              <li>
                Conducted common-mode emissions on the motor cable, measured at a
                50 Ω LISN with a 9 kHz resolution bandwidth.
              </li>
              <li>
                Three bands: 150 kHz – 500 kHz, 500 kHz – 5 MHz and 5 MHz – 30 MHz.
              </li>
              <li>
                Compared against a synthetic EN 12016-style limit curve. The curve
                is an assumption, not published data.
              </li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  )
}
