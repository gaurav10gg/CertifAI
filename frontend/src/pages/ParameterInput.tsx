import { useState } from 'react'

import type {
  DeviceParameters,
  ParameterSpec,
  PwmModulationType,
  PwmOption,
  SchematicImportResult,
  SwitchingDeviceOption,
  SwitchingDeviceType,
} from '../api/types'
import { Card, Eyebrow } from '../components/Card'
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  SpinnerIcon,
} from '../components/Icons'
import { SchematicFindings } from '../components/SchematicFindings'
import { SegmentedControl, Slider } from '../components/Slider'

/**
 * ParameterInput.tsx -- stage two: the design parameters.
 *
 * Bounds, steps and tooltip text all come from the backend
 * (PARAMETER_RANGES in simulate.py), so the UI cannot drift out of the range the
 * model was trained on. Switching frequency is 3–16 kHz, standard VFD practice.
 */

/** Percentage-style parameters are stored 0-1 but are far easier to read as %. */
const PERCENT_KEYS = new Set(['shielding_quality', 'input_filter_quality'])
const CM_CHOKE_MAX_MH = 2

const POWER_STAGE_KEYS = [
  'switching_frequency_khz',
  'dv_dt_v_per_us',
  'di_dt_a_per_us',
  'cable_length_m',
  'shielding_quality',
  'load_current_a',
  'cm_choke_effectiveness',
  'input_filter_quality',
] as const

const DEVICE_TOOLTIP =
  'GaN switches about 5× faster than silicon at the same rated edge speed — this changes predicted emissions without moving the dv/dt slider itself.'

export function ParameterInput({
  specs,
  pwmOptions,
  deviceOptions,
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
  schematic = null,
}: {
  specs: ParameterSpec[]
  pwmOptions: PwmOption[]
  deviceOptions: SwitchingDeviceOption[]
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
  schematic?: SchematicImportResult | null
}) {
  const setNumeric = (key: ParameterSpec['key'], value: number) =>
    onChange({ ...parameters, [key]: value })

  const byKey = new Map(specs.map((spec) => [spec.key, spec]))
  const powerSpecs = POWER_STAGE_KEYS.map((key) => byKey.get(key)).filter(
    (spec): spec is ParameterSpec => spec !== undefined,
  )
  const clockSpec = byKey.get('clock_frequency_mhz')
  const [advancedOpen, setAdvancedOpen] = useState(false)

  return (
    <div className="motion-safe:animate-fade-up">
      <header className="max-w-2xl">
        <Eyebrow>Step 2</Eyebrow>
        <h1 className="mt-2 text-display font-semibold tracking-tight text-ink">
          {isCustom ? 'Custom configuration' : deviceName}
        </h1>
        <p className="mt-3 text-base leading-relaxed text-ink-muted">
          {schematic
            ? 'The hardware below was read from your schematic. Fill in the four installation and firmware values it cannot carry, then run the assessment.'
            : 'These parameters drive the physics simulation. Hover any label to see what the quantity represents. Switching frequency is limited to 3–16 kHz.'}
        </p>
      </header>

      {schematic ? (
        <div className="mt-9">
          <SchematicFindings report={schematic} specs={specs} />
        </div>
      ) : null}

      <div className="mt-9 grid gap-6 lg:grid-cols-[1.55fr_1fr] lg:items-start">
        <div className="space-y-6">
        <Card className="space-y-7">
          <div>
            <Eyebrow>Power stage</Eyebrow>
            <p className="mt-1 text-sm text-ink-muted">
              These controls feed the conducted common-mode model, except di/dt,
              which is the magnetic source for the radiated score.
            </p>
          </div>
          {powerSpecs.map((spec) => {
            const isPercent = PERCENT_KEYS.has(spec.key)
            const isChoke = spec.key === 'cm_choke_effectiveness'
            const scale = isPercent ? 100 : isChoke ? CM_CHOKE_MAX_MH : 1
            const needsYou = Boolean(schematic?.missing.includes(spec.key))
            const baseLabel = isPercent
              ? `${spec.label} (%)`
              : isChoke
                ? 'Common-mode choke'
                : spec.label
            return (
              <Slider
                key={spec.key}
                label={needsYou ? `${baseLabel} · needed` : baseLabel}
                unit={isPercent ? '%' : isChoke ? 'mH' : spec.unit}
                min={spec.min * scale}
                max={spec.max * scale}
                step={isChoke ? 0.1 : spec.step * scale}
                value={(parameters[spec.key] as number) * scale}
                precision={isPercent ? 0 : isChoke ? 1 : undefined}
                description={spec.description}
                onChange={(value) => setNumeric(spec.key, value / scale)}
              />
            )
          })}

          <div className="space-y-7 border-t border-line pt-7">
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
            <SegmentedControl<SwitchingDeviceType>
              label="Switching device type"
              description={DEVICE_TOOLTIP}
              options={deviceOptions.map((option) => ({
                value: option.value,
                label: option.label,
                hint: `×${option.edge_multiplier} edge`,
              }))}
              value={parameters.switching_device_type}
              onChange={(value) =>
                onChange({ ...parameters, switching_device_type: value })
              }
            />
          </div>
        </Card>

        {clockSpec ? (
          <Card className="space-y-4 border-dashed">
            <div>
              <Eyebrow>Digital control electronics</Eyebrow>
              <p className="mt-1 text-sm text-ink-muted">
                This clock does not enter the conducted common-mode model. It is
                a source for the separate radiated-emissions score.
              </p>
            </div>
            <Slider
              label={clockSpec.label}
              unit={clockSpec.unit}
              min={clockSpec.min}
              max={clockSpec.max}
              step={clockSpec.step}
              value={parameters.clock_frequency_mhz}
              description={clockSpec.description}
              onChange={(value) => setNumeric('clock_frequency_mhz', value)}
            />
          </Card>
        ) : null}

        <Card className="border-dashed">
          <button
            type="button"
            className="flex w-full items-center justify-between gap-3 text-left"
            aria-expanded={advancedOpen}
            onClick={() => setAdvancedOpen((open) => !open)}
          >
            <span>
              <Eyebrow>Optional</Eyebrow>
              <span className="mt-1 block text-base font-semibold text-ink">
                Additional Parameters (Advanced)
              </span>
            </span>
            <span className="text-sm text-ink-muted">{advancedOpen ? 'Hide' : 'Show'}</span>
          </button>
          {advancedOpen ? (
            <div className="mt-5 space-y-7 border-t border-dashed border-line pt-5">
              <p className="text-sm leading-relaxed text-ink">
                These parameters shape the simulated waveforms and emission spectrum
                shown below, and affect the full-detail prediction. They are
                validated at the simulator level, but are not yet named inputs to
                the explainable risk model above — the Why This Margin chart and
                What To Change Next recommendations will not mention these by name.
                Promoting them into the explainable model would require retraining
                and re-validating that model family, which is a larger step than
                this pass covers.
              </p>

              <div className="space-y-6">
                <Eyebrow>Bus and filtering</Eyebrow>
                <Slider
                  label="DC bus voltage"
                  unit="V"
                  min={400}
                  max={800}
                  step={5}
                  value={parameters.dc_bus_voltage_v}
                  description="Height of the DC bus. The common-mode, DC-link and motor-voltage waveforms scale with it. 565 V is the rectified 400 V assumption used everywhere else in this tool."
                  onChange={(value) => onChange({ ...parameters, dc_bus_voltage_v: value })}
                />
                <Slider
                  label="Bus capacitor ESL"
                  unit="nH"
                  min={0}
                  max={500}
                  step={10}
                  value={parameters.dc_bus_esl_h * 1e9}
                  precision={0}
                  description="Parasitic bus inductance — causes ringing after each switching transition. Zero means an ideal bus, with no ring."
                  onChange={(value) => onChange({ ...parameters, dc_bus_esl_h: value * 1e-9 })}
                />
                <Slider
                  label="Bus capacitor ESR"
                  unit="mΩ"
                  min={0}
                  max={200}
                  step={5}
                  value={parameters.dc_bus_esr_ohm * 1e3}
                  precision={0}
                  description="Series resistance of that same parasitic path. It damps the ring. It does nothing while ESL is zero."
                  onChange={(value) => onChange({ ...parameters, dc_bus_esr_ohm: value * 1e-3 })}
                />
                <Slider
                  label="DC-link choke"
                  unit="mH"
                  min={0}
                  max={5}
                  step={0.1}
                  value={parameters.dc_link_choke_h * 1e3}
                  description="Series inductor on the DC bus. It smooths bus ripple. It is not the common-mode choke, which acts on the motor-cable current instead."
                  onChange={(value) => onChange({ ...parameters, dc_link_choke_h: value * 1e-3 })}
                />
                <Slider
                  label="Y-capacitor"
                  unit="nF"
                  min={0}
                  max={100}
                  step={1}
                  value={parameters.y_capacitance_f * 1e9}
                  precision={0}
                  description="A ground-shunt filter capacitor. It attenuates common-mode noise around its self-resonance, then the attenuation eases off because a real Y-cap has lead inductance. Separate from the common-mode choke: a choke blocks the current, a Y-cap shunts it. The model includes 15 nH and 2 Ω of parasitic impedance so the dip is finite, not a perfect short."
                  onChange={(value) => onChange({ ...parameters, y_capacitance_f: value * 1e-9 })}
                />
              </div>

              <div className="space-y-6 border-t border-line pt-6">
                <Eyebrow>Modulation and rectification</Eyebrow>
                <Slider
                  label="Dead time"
                  unit="µs"
                  min={0}
                  max={5}
                  step={0.1}
                  value={parameters.dead_time_us}
                  description="Blanking time between the two devices in a leg. The volt-second error follows the current direction and adds a small amount of low-order distortion. Zero leaves the PWM edges untouched."
                  onChange={(value) => onChange({ ...parameters, dead_time_us: value })}
                />
                <div>
                  <label className="flex items-center justify-between gap-3 text-sm font-medium text-ink">
                    <span>Spread-spectrum carrier</span>
                    <input
                      type="checkbox"
                      className="h-4 w-4 accent-current"
                      checked={parameters.spread_spectrum}
                      onChange={(event) =>
                        onChange({ ...parameters, spread_spectrum: event.target.checked })
                      }
                    />
                  </label>
                  <p className="mt-1 text-xs leading-relaxed text-ink-muted">
                    Dithers the switching period slightly each cycle. Spreads
                    concentrated peak energy into a wider band — reduces how far
                    the peak stands above its neighbours, even when the tallest
                    max-hold bin does not fall.
                  </p>
                </div>
                {parameters.spread_spectrum ? (
                  <Slider
                    label="Carrier dither"
                    unit="%"
                    min={2}
                    max={35}
                    step={1}
                    value={parameters.spread_spectrum_jitter * 100}
                    precision={0}
                    description="How far the carrier period is allowed to wander, as a percentage either side of the set switching frequency. The default is ±10%."
                    onChange={(value) =>
                      onChange({ ...parameters, spread_spectrum_jitter: value / 100 })
                    }
                  />
                ) : null}
                <SegmentedControl<'DIODE_6PULSE' | 'ACTIVE_FRONT_END'>
                  label="Rectifier type"
                  description="6-pulse diode bridge: the usual 5th and 7th harmonic current. Active front end: a near-sinusoidal input current with much lower distortion."
                  options={[
                    { value: 'DIODE_6PULSE', label: '6-pulse diode' },
                    { value: 'ACTIVE_FRONT_END', label: 'Active front end' },
                  ]}
                  value={parameters.rectifier_type}
                  onChange={(value) => onChange({ ...parameters, rectifier_type: value })}
                />
              </div>
            </div>
          ) : null}
        </Card>
        </div>

        <div className="space-y-4 lg:sticky lg:top-20">
          <Card>
            <Eyebrow>Run the risk assessment</Eyebrow>
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">
              The backend synthesises five waveforms, applies a receiver-emulated
              FFT across the three EMC bands, and evaluates the monotonically
              constrained models. This takes about a second. The result is a
              pre-compliance risk indicator, not a certification prediction.
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
                  Run risk assessment
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
                Conducted common-mode emissions on the motor cable, 150 kHz–30 MHz,
                measured at a 50 Ω LISN — the main risk score.
              </li>
              <li>
                A separate radiated estimate, 30 MHz–1 GHz, from the clock and
                di/dt. That score is exploratory and is not mixed into the
                conducted result.
              </li>
              <li>
                Motor and input current THD as a separate power-quality readout.
                Input filter quality moves THD, not the emission bands.
              </li>
              <li>
                Compared against a synthetic EN 12016-style limit curve. The curve
                is an assumption, not published data, and this is not a
                certification prediction.
              </li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  )
}
