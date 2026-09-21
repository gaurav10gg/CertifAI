import { useState } from 'react'

import { ApiError, downloadCertificate } from '../api/client'
import type { PredictionResult } from '../api/types'
import { CheckIcon, DownloadIcon, SpinnerIcon } from './Icons'

/**
 * CertificateButton.tsx -- requests and downloads the assessment PDF.
 *
 * Only the parameters are sent; the backend recomputes the assessment before
 * rendering, so the document can never assert a score the model did not produce.
 * The button reports its own progress and any failure inline rather than through
 * an alert, since a failed download is recoverable by retrying.
 */

type Status = 'idle' | 'working' | 'done' | 'error'

export function CertificateButton({ result }: { result: PredictionResult }) {
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)

  const onClick = async () => {
    setStatus('working')
    setError(null)
    try {
      await downloadCertificate({
        parameters: result.parameters,
        device_id: result.device_id,
        device_name: result.device_name,
      })
      setStatus('done')
      window.setTimeout(() => setStatus('idle'), 2600)
    } catch (caught) {
      setStatus('error')
      setError(
        caught instanceof ApiError
          ? caught.message
          : 'The pre-compliance report could not be generated.',
      )
    }
  }

  return (
    <div className="flex flex-col items-start gap-2">
      <button
        type="button"
        onClick={onClick}
        disabled={status === 'working'}
        className="btn-primary w-full sm:w-auto"
      >
        {status === 'working' ? (
          <>
            <SpinnerIcon className="h-4 w-4" />
            Generating report…
          </>
        ) : status === 'done' ? (
          <>
            <CheckIcon className="h-4 w-4" />
            Downloaded
          </>
        ) : (
          <>
            <DownloadIcon className="h-4 w-4" />
            Download report
          </>
        )}
      </button>

      {error ? <p className="text-xs text-fail">{error}</p> : null}
    </div>
  )
}
