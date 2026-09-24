import { useRef, useState } from 'react'

import { ApiError, importSchematic } from '../api/client'
import type { SchematicImportResult } from '../api/types'
import { SpinnerIcon, UploadIcon } from './Icons'

/**
 * SchematicUpload.tsx -- third entry path on the select page.
 *
 * A schematic PDF gives the hardware (devices, gate resistors, CM chokes,
 * Y-caps). It does not give the carrier, cable, shielding or load, so the
 * result lands on the parameter page half-filled with the rest asked for.
 */

export function SchematicUpload({
  onImported,
}: {
  onImported: (report: SchematicImportResult) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)

  const handle = async (file: File | undefined) => {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const report = await importSchematic(file)
      if (!report.ok) {
        setError(report.reason ?? 'This PDF could not be read.')
        return
      }
      onImported(report)
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Upload failed.')
    } finally {
      setBusy(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        void handle(event.dataTransfer.files?.[0])
      }}
      className={`group flex min-h-[200px] flex-col items-start justify-center rounded-card
        border border-dashed p-5 text-left transition-all duration-200 ease-standard
        ${dragging ? 'border-accent bg-surface' : 'border-line-strong bg-transparent hover:border-accent hover:bg-surface'}`}
    >
      <UploadIcon className="h-6 w-6 text-ink-muted transition-colors group-hover:text-ink" />
      <h2 className="mt-4 text-base font-semibold tracking-tight text-ink">
        Upload a schematic PDF
      </h2>
      <p className="mt-1.5 text-sm leading-relaxed text-ink-muted">
        Reads devices, gate resistors, common-mode chokes and Y-capacitors from
        the drawing's text layer, then asks for what a schematic cannot show.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          className="btn-secondary"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
        >
          {busy ? <SpinnerIcon className="h-4 w-4" /> : null}
          {busy ? 'Reading…' : 'Choose PDF'}
        </button>
        <span className="text-2xs text-ink-faint">or drop it here · CAD export with text, not a scan</span>
      </div>
      {error ? <p className="mt-3 text-xs text-fail">{error}</p> : null}
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        className="hidden"
        onChange={(event) => void handle(event.target.files?.[0])}
      />
    </div>
  )
}
