import type {
  DeviceParameters,
  DevicesResponse,
  HealthResponse,
  MethodologyResponse,
  PredictionResult,
} from './types'

/**
 * client.ts -- thin fetch wrapper for the CertifAI API.
 *
 * Paths are relative: Vite proxies /api to the backend in development (see
 * vite.config.ts), and in production both are served from the same origin.
 */

const BASE = '/api'

export class ApiError extends Error {
  readonly status: number
  /** True for 503, which the backend returns when the model artifacts are absent. */
  readonly isModelMissing: boolean

  constructor(message: string, status: number, isModelMissing = false) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.isModelMissing = isModelMissing
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch {
    throw new ApiError(
      'Could not reach the CertifAI backend. Confirm it is running on port 8000.',
      0,
    )
  }

  if (!response.ok) {
    const detail = await extractDetail(response)
    throw new ApiError(detail, response.status, response.status === 503)
  }

  return (await response.json()) as T
}

async function extractDetail(response: Response): Promise<string> {
  try {
    const body = await response.json()
    const detail = body?.detail
    if (typeof detail === 'string') return detail
    // FastAPI validation errors arrive as an array of per-field objects.
    if (Array.isArray(detail)) {
      return detail
        .map((item: { loc?: string[]; msg?: string }) =>
          [item.loc?.slice(1).join('.'), item.msg].filter(Boolean).join(': '),
        )
        .join('; ')
    }
  } catch {
    /* fall through to the generic message */
  }
  return `Request failed with status ${response.status}.`
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function fetchDevices(): Promise<DevicesResponse> {
  return request<DevicesResponse>('/devices')
}

export function fetchMethodology(): Promise<MethodologyResponse> {
  return request<MethodologyResponse>('/methodology')
}

export function runPrediction(payload: {
  parameters: DeviceParameters
  device_id?: string | null
  device_name?: string | null
}): Promise<PredictionResult> {
  return request<PredictionResult>('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

/**
 * Download the assessment PDF.
 *
 * The backend recomputes the assessment from the parameters, so only those are
 * sent. The filename comes from the Content-Disposition header the server sets.
 */
export async function downloadCertificate(payload: {
  parameters: DeviceParameters
  device_id?: string | null
  device_name?: string | null
}): Promise<void> {
  const response = await fetch(`${BASE}/certificate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    throw new ApiError(await extractDetail(response), response.status)
  }

  const blob = await response.blob()
  const filename =
    parseFilename(response.headers.get('Content-Disposition')) ??
    'certifai-pre-compliance-assessment.pdf'

  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function parseFilename(header: string | null): string | null {
  if (!header) return null
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(header)
  return match ? decodeURIComponent(match[1]) : null
}
