import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  ApiError,
  fetchDevices,
  fetchMethodology,
  fetchTradeoff,
  fetchValidation,
  runPrediction,
} from './api/client'
import type {
  AssessmentHistoryEntry,
  DeviceParameters,
  DeviceProfile,
  DevicesResponse,
  MethodologyResponse,
  PredictionResult,
  SchematicImportResult,
  TradeoffResponse,
  ValidationResponse,
} from './api/types'
import { Disclaimer } from './components/Disclaimer'
import { Header } from './components/Header'
import { Modal } from './components/Modal'
import { Stepper, type StepId } from './components/Stepper'
import { DeviceSelect } from './pages/DeviceSelect'
import { Methodology } from './pages/Methodology'
import { ModelValidation } from './pages/ModelValidation'
import { ParameterInput } from './pages/ParameterInput'
import { Results } from './pages/Results'
import { useTheme } from './theme/useTheme'

function heldParameterKey(parameters: DeviceParameters): string {
  const { switching_frequency_khz: _carrier, ...held } = parameters
  return JSON.stringify(held)
}

/**
 * App.tsx -- flow state and data loading.
 *
 * Three stages, held in local state rather than a router. Session history of
 * the last three assessments is kept for the "What changed" sparkline.
 */

function defaultParameters(devices: DevicesResponse): DeviceParameters {
  const numeric = Object.fromEntries(
    devices.parameters.map((spec) => [spec.key, spec.default]),
  ) as Record<string, number>

  return {
    switching_frequency_khz: numeric.switching_frequency_khz ?? 8,
    dv_dt_v_per_us: numeric.dv_dt_v_per_us ?? 3500,
    cable_length_m: numeric.cable_length_m ?? 25,
    shielding_quality: numeric.shielding_quality ?? 0.6,
    load_current_a: numeric.load_current_a ?? 45,
    pwm_modulation_type: 'SPWM',
    switching_device_type: 'SI_IGBT',
    input_filter_quality: numeric.input_filter_quality ?? 0.45,
    cm_choke_effectiveness: numeric.cm_choke_effectiveness ?? 0,
    clock_frequency_mhz: numeric.clock_frequency_mhz ?? 48,
    di_dt_a_per_us: numeric.di_dt_a_per_us ?? 200,
    dc_bus_voltage_v: 565,
    dc_bus_esl_h: 0,
    dc_bus_esr_ohm: 0,
    dead_time_us: 0,
    spread_spectrum: false,
    spread_spectrum_jitter: 0.1,
    rectifier_type: 'DIODE_6PULSE',
    dc_link_choke_h: 0,
    y_capacitance_f: 0,
  }
}

function sameParameters(a: DeviceParameters, b: DeviceParameters): boolean {
  return (Object.keys(a) as (keyof DeviceParameters)[]).every(
    (key) => a[key] === b[key],
  )
}

function withFilterDefault(parameters: DeviceParameters): DeviceParameters {
  return {
    ...parameters,
    input_filter_quality: parameters.input_filter_quality ?? 0.45,
    cm_choke_effectiveness: parameters.cm_choke_effectiveness ?? 0,
    clock_frequency_mhz: parameters.clock_frequency_mhz ?? 48,
    di_dt_a_per_us: parameters.di_dt_a_per_us ?? 200,
    switching_device_type: parameters.switching_device_type ?? 'SI_IGBT',
    dc_bus_voltage_v: parameters.dc_bus_voltage_v ?? 565,
    dc_bus_esl_h: parameters.dc_bus_esl_h ?? 0,
    dc_bus_esr_ohm: parameters.dc_bus_esr_ohm ?? 0,
    dead_time_us: parameters.dead_time_us ?? 0,
    spread_spectrum: parameters.spread_spectrum ?? false,
    spread_spectrum_jitter: parameters.spread_spectrum_jitter ?? 0.1,
    rectifier_type: parameters.rectifier_type ?? 'DIODE_6PULSE',
    dc_link_choke_h: parameters.dc_link_choke_h ?? 0,
    y_capacitance_f: parameters.y_capacitance_f ?? 0,
  }
}

export default function App() {
  const { theme, toggle } = useTheme()

  const [catalogue, setCatalogue] = useState<DevicesResponse | null>(null)
  const [catalogueError, setCatalogueError] = useState<string | null>(null)

  const [step, setStep] = useState<StepId>('select')
  const [furthest, setFurthest] = useState<StepId>('select')
  const [device, setDevice] = useState<DeviceProfile | null>(null)
  const [parameters, setParameters] = useState<DeviceParameters | null>(null)

  const [result, setResult] = useState<PredictionResult | null>(null)
  const [schematic, setSchematic] = useState<SchematicImportResult | null>(null)
  const [history, setHistory] = useState<AssessmentHistoryEntry[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [predictError, setPredictError] = useState<string | null>(null)

  const [tradeoff, setTradeoff] = useState<TradeoffResponse | null>(null)
  const [tradeoffLoading, setTradeoffLoading] = useState(false)
  const [tradeoffError, setTradeoffError] = useState<string | null>(null)
  const tradeoffKeyRef = useRef<string | null>(null)
  const tradeoffRequestRef = useRef(0)
  const [applyingCarrier, setApplyingCarrier] = useState(false)
  const [applyingFrequency, setApplyingFrequency] = useState<number | null>(null)

  const [methodologyOpen, setMethodologyOpen] = useState(false)
  const [methodology, setMethodology] = useState<MethodologyResponse | null>(null)
  const [methodologyLoading, setMethodologyLoading] = useState(false)
  const [methodologyError, setMethodologyError] = useState<string | null>(null)

  const [validationOpen, setValidationOpen] = useState(false)
  const [validation, setValidation] = useState<ValidationResponse | null>(null)
  const [validationLoading, setValidationLoading] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchDevices()
      .then((data) => {
        if (!cancelled) setCatalogue(data)
      })
      .catch((caught: unknown) => {
        if (cancelled) return
        setCatalogueError(
          caught instanceof ApiError
            ? caught.message
            : 'Could not reach the EMC Advisor backend. Is it running on port 8000?',
        )
      })
    return () => {
      cancelled = true
    }
  }, [])

  const openMethodology = useCallback(() => {
    setMethodologyOpen(true)
    if (methodology || methodologyLoading) return

    setMethodologyLoading(true)
    setMethodologyError(null)
    fetchMethodology()
      .then(setMethodology)
      .catch((caught: unknown) =>
        setMethodologyError(
          caught instanceof ApiError
            ? caught.message
            : 'Could not load the methodology document.',
        ),
      )
      .finally(() => setMethodologyLoading(false))
  }, [methodology, methodologyLoading])

  const openValidation = useCallback(() => {
    setValidationOpen(true)
    if (validation || validationLoading) return

    setValidationLoading(true)
    setValidationError(null)
    fetchValidation()
      .then(setValidation)
      .catch((caught: unknown) =>
        setValidationError(
          caught instanceof ApiError
            ? caught.message
            : 'Could not load the validation suite.',
        ),
      )
      .finally(() => setValidationLoading(false))
  }, [validation, validationLoading])

  const advance = useCallback((next: StepId) => {
    const order: StepId[] = ['select', 'configure', 'results']
    setStep(next)
    setFurthest((current) =>
      order.indexOf(next) > order.indexOf(current) ? next : current,
    )
  }, [])

  const clearTradeoff = () => {
    tradeoffKeyRef.current = null
    tradeoffRequestRef.current += 1
    setTradeoff(null)
    setTradeoffLoading(false)
    setTradeoffError(null)
  }

  const requestTradeoff = (params: DeviceParameters) => {
    const next = withFilterDefault(params)
    const key = heldParameterKey(next)
    if (key === tradeoffKeyRef.current) return
    tradeoffKeyRef.current = key
    const requestId = ++tradeoffRequestRef.current
    setTradeoffLoading(true)
    setTradeoffError(null)
    fetchTradeoff({ parameters: next })
      .then((data) => {
        if (requestId !== tradeoffRequestRef.current) return
        setTradeoff(data)
      })
      .catch((caught: unknown) => {
        if (requestId !== tradeoffRequestRef.current) return
        setTradeoff(null)
        setTradeoffError(
          caught instanceof ApiError
            ? caught.message
            : 'The trade-off sweep failed. Please try again.',
        )
      })
      .finally(() => {
        if (requestId !== tradeoffRequestRef.current) return
        setTradeoffLoading(false)
      })
  }

  const onSelectDevice = (profile: DeviceProfile) => {
    setDevice(profile)
    setSchematic(null)
    setParameters(withFilterDefault(profile.parameters))
    setResult(null)
    setPredictError(null)
    clearTradeoff()
    advance('configure')
  }

  const onCustom = () => {
    if (!catalogue) return
    setDevice(null)
    setSchematic(null)
    setParameters(defaultParameters(catalogue))
    setResult(null)
    setPredictError(null)
    clearTradeoff()
    advance('configure')
  }

  const onSchematic = (report: SchematicImportResult) => {
    if (!catalogue) return
    // Start from mid-range defaults for the fields a schematic cannot carry,
    // then overwrite with whatever the drawing gave us.
    const base = defaultParameters(catalogue)
    const merged: DeviceParameters = { ...base, ...(report.parameters ?? {}) } as DeviceParameters
    setDevice(null)
    setSchematic(report)
    setParameters(withFilterDefault(merged))
    setResult(null)
    setPredictError(null)
    clearTradeoff()
    advance('configure')
  }

  const onRestart = () => {
    setStep('select')
    setFurthest('select')
    setDevice(null)
    setSchematic(null)
    setParameters(null)
    setResult(null)
    setPredictError(null)
    clearTradeoff()
  }

  const recordHistory = (prediction: PredictionResult) => {
    setHistory((current) =>
      [
        {
          score: prediction.risk_score,
          plusMinus: prediction.risk_score_plus_minus,
          level: prediction.risk_level,
          name: prediction.device_name,
          at: prediction.generated_at,
        },
        ...current,
      ].slice(0, 3),
    )
  }

  const onSubmit = async () => {
    if (!parameters) return
    setSubmitting(true)
    setPredictError(null)
    requestTradeoff(parameters)
    try {
      const prediction = await runPrediction({
        parameters: withFilterDefault(parameters),
        device_id: device?.id ?? null,
        device_name: device?.name,
      })
      setResult(prediction)
      recordHistory(prediction)
      advance('results')
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (caught) {
      setPredictError(
        caught instanceof ApiError
          ? caught.message
          : 'The risk assessment failed. Please try again.',
      )
    } finally {
      setSubmitting(false)
    }
  }

  const onApplyCarrier = async (khz: number) => {
    if (!parameters || applyingCarrier) return
    const next = {
      ...withFilterDefault(parameters),
      switching_frequency_khz: khz,
    }
    if (
      result &&
      Math.abs(next.switching_frequency_khz - result.parameters.switching_frequency_khz) <
        0.05
    ) {
      return
    }
    setParameters(next)
    setApplyingCarrier(true)
    setApplyingFrequency(khz)
    setPredictError(null)
    try {
      const prediction = await runPrediction({
        parameters: next,
        device_id: device?.id ?? null,
        device_name: device?.name,
      })
      setResult(prediction)
      recordHistory(prediction)
    } catch (caught) {
      setPredictError(
        caught instanceof ApiError
          ? caught.message
          : 'The risk assessment failed. Please try again.',
      )
    } finally {
      setApplyingCarrier(false)
      setApplyingFrequency(null)
    }
  }

  const dirty = useMemo(
    () =>
      device !== null &&
      parameters !== null &&
      !sameParameters(withFilterDefault(device.parameters), parameters),
    [device, parameters],
  )

  return (
    <div className="min-h-screen bg-paper">
      <Header
        theme={theme}
        onToggleTheme={toggle}
        onOpenMethodology={openMethodology}
        onOpenValidation={openValidation}
        onReset={onRestart}
        canReset={step !== 'select'}
      />

      <main className="mx-auto max-w-6xl px-5 pb-20 pt-8 sm:px-8 sm:pt-10">
        <div className="mb-8">
          <Stepper
            current={step}
            furthest={furthest}
            onNavigate={(target) => {
              if (target === 'results' && !result) return
              setStep(target)
            }}
          />
        </div>

        {catalogueError ? (
          <BackendUnavailable message={catalogueError} />
        ) : step === 'select' || !catalogue || !parameters ? (
          <DeviceSelect
            devices={catalogue?.devices ?? []}
            selectedId={device?.id ?? null}
            onSelect={onSelectDevice}
            onCustom={onCustom}
            onSchematic={onSchematic}
            loading={!catalogue}
          />
        ) : step === 'configure' ? (
          <ParameterInput
            specs={catalogue.parameters}
            pwmOptions={catalogue.pwm_modulation_types}
            deviceOptions={catalogue.switching_device_types ?? []}
            parameters={parameters}
            deviceName={device?.name ?? schematic?.title ?? 'Custom configuration'}
            isCustom={device === null && schematic === null}
            schematic={schematic}
            onChange={setParameters}
            onBack={() => setStep('select')}
            onSubmit={onSubmit}
            onResetToPreset={() =>
              device && setParameters(withFilterDefault(device.parameters))
            }
            dirty={dirty}
            submitting={submitting}
            error={predictError}
          />
        ) : result ? (
          <Results
            result={result}
            history={history}
            specs={catalogue.parameters}
            devices={catalogue.devices}
            tradeoff={tradeoff}
            tradeoffLoading={tradeoffLoading}
            tradeoffError={tradeoffError}
            applyingCarrier={applyingCarrier}
            applyingFrequency={applyingFrequency}
            applyError={predictError}
            onApplyCarrier={onApplyCarrier}
            onBack={() => setStep('configure')}
            onRestart={onRestart}
            onOpenMethodology={openMethodology}
            onOpenValidation={openValidation}
            schematic={schematic}
          />
        ) : null}

        {step !== 'results' && catalogue ? (
          <Disclaimer
            text={catalogue.disclaimer}
            onOpenMethodology={openMethodology}
            className="mt-14 border-t border-line pt-5"
          />
        ) : null}
      </main>

      <Modal
        open={methodologyOpen}
        onClose={() => setMethodologyOpen(false)}
        title="How EMC Advisor works"
      >
        <Methodology
          data={methodology}
          loading={methodologyLoading}
          error={methodologyError}
        />
      </Modal>

      <Modal
        open={validationOpen}
        onClose={() => setValidationOpen(false)}
        title="Model validation"
      >
        <ModelValidation
          data={validation}
          loading={validationLoading}
          error={validationError}
        />
      </Modal>
    </div>
  )
}

function BackendUnavailable({ message }: { message: string }) {
  return (
    <div className="mx-auto max-w-lg rounded-card border border-line bg-surface p-7 text-center">
      <h1 className="text-lg font-semibold tracking-tight text-ink">
        Backend unavailable
      </h1>
      <p className="mt-2 text-sm leading-relaxed text-ink-muted">{message}</p>
      <pre className="mt-4 overflow-x-auto rounded-md border border-line bg-paper px-3 py-2.5 text-left text-2xs text-ink-muted">
        cd backend{'\n'}uvicorn app:app --port 8000
      </pre>
      <button
        type="button"
        onClick={() => window.location.reload()}
        className="btn-secondary mt-5"
      >
        Retry
      </button>
    </div>
  )
}
