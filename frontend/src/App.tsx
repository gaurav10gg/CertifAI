import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  ApiError,
  fetchDevices,
  fetchMethodology,
  runPrediction,
} from './api/client'
import type {
  DeviceParameters,
  DeviceProfile,
  DevicesResponse,
  MethodologyResponse,
  PredictionResult,
} from './api/types'
import { Disclaimer } from './components/Disclaimer'
import { Header } from './components/Header'
import { Modal } from './components/Modal'
import { Stepper, type StepId } from './components/Stepper'
import { DeviceSelect } from './pages/DeviceSelect'
import { Methodology } from './pages/Methodology'
import { ParameterInput } from './pages/ParameterInput'
import { Results } from './pages/Results'
import { useTheme } from './theme/useTheme'

/**
 * App.tsx -- flow state and data loading.
 *
 * Three stages, held in local state rather than a router: the flow is short and
 * linear, and a deep link to a results page would be meaningless without the
 * parameters that produced it. The user may step backwards freely; results are
 * kept so returning from a parameter edit does not lose the previous assessment.
 *
 * Methodology is a sheet rather than a stage, so it can be read at any point
 * without abandoning work in progress.
 */

/** Mid-range starting point for a custom configuration, built from the backend specs. */
function defaultParameters(devices: DevicesResponse): DeviceParameters {
  const numeric = Object.fromEntries(
    devices.parameters.map((spec) => [spec.key, spec.default]),
  ) as Omit<DeviceParameters, 'pwm_modulation_type'>

  return { ...numeric, pwm_modulation_type: 'SPWM' }
}

function sameParameters(a: DeviceParameters, b: DeviceParameters): boolean {
  return (Object.keys(a) as (keyof DeviceParameters)[]).every(
    (key) => a[key] === b[key],
  )
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
  const [submitting, setSubmitting] = useState(false)
  const [predictError, setPredictError] = useState<string | null>(null)

  const [methodologyOpen, setMethodologyOpen] = useState(false)
  const [methodology, setMethodology] = useState<MethodologyResponse | null>(null)
  const [methodologyLoading, setMethodologyLoading] = useState(false)
  const [methodologyError, setMethodologyError] = useState<string | null>(null)

  // Device catalogue and parameter specs. Everything downstream depends on this.
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
            : 'Could not reach the CertifAI backend. Is it running on port 8000?',
        )
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Methodology is fetched lazily, on the first open, from the event that opens
  // it rather than from an effect watching the open flag.
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

  const advance = useCallback((next: StepId) => {
    const order: StepId[] = ['select', 'configure', 'results']
    setStep(next)
    setFurthest((current) =>
      order.indexOf(next) > order.indexOf(current) ? next : current,
    )
  }, [])

  const onSelectDevice = (profile: DeviceProfile) => {
    setDevice(profile)
    setParameters(profile.parameters)
    setResult(null)
    setPredictError(null)
    advance('configure')
  }

  const onCustom = () => {
    if (!catalogue) return
    setDevice(null)
    setParameters(defaultParameters(catalogue))
    setResult(null)
    setPredictError(null)
    advance('configure')
  }

  const onRestart = () => {
    setStep('select')
    setFurthest('select')
    setDevice(null)
    setParameters(null)
    setResult(null)
    setPredictError(null)
  }

  const onSubmit = async () => {
    if (!parameters) return
    setSubmitting(true)
    setPredictError(null)
    try {
      const prediction = await runPrediction({
        parameters,
        device_id: device?.id ?? null,
        device_name: device?.name,
      })
      setResult(prediction)
      advance('results')
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (caught) {
      setPredictError(
        caught instanceof ApiError
          ? caught.message
          : 'The compliance check failed. Please try again.',
      )
    } finally {
      setSubmitting(false)
    }
  }

  const dirty = useMemo(
    () =>
      device !== null &&
      parameters !== null &&
      !sameParameters(device.parameters, parameters),
    [device, parameters],
  )

  return (
    <div className="min-h-screen bg-paper">
      <Header
        theme={theme}
        onToggleTheme={toggle}
        onOpenMethodology={openMethodology}
        onReset={onRestart}
        canReset={step !== 'select'}
      />

      <main className="mx-auto max-w-6xl px-5 pb-20 pt-8 sm:px-8 sm:pt-10">
        <div className="mb-8">
          <Stepper
            current={step}
            furthest={furthest}
            onNavigate={(target) => {
              // Results are only reachable again if an assessment exists.
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
            loading={!catalogue}
          />
        ) : step === 'configure' ? (
          <ParameterInput
            specs={catalogue.parameters}
            pwmOptions={catalogue.pwm_modulation_types}
            parameters={parameters}
            deviceName={device?.name ?? 'Custom configuration'}
            isCustom={device === null}
            onChange={setParameters}
            onBack={() => setStep('select')}
            onSubmit={onSubmit}
            onResetToPreset={() => device && setParameters(device.parameters)}
            dirty={dirty}
            submitting={submitting}
            error={predictError}
          />
        ) : result ? (
          <Results
            result={result}
            onBack={() => setStep('configure')}
            onRestart={onRestart}
            onOpenMethodology={openMethodology}
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
        title="How CertifAI works"
      >
        <Methodology
          data={methodology}
          loading={methodologyLoading}
          error={methodologyError}
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
