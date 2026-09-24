import { useState } from 'react'

import type { ParameterSpec, SchematicImportResult } from '../api/types'
import { Card, Eyebrow } from './Card'

/**
 * SchematicFindings.tsx -- what was read from the uploaded PDF, and what
 * still has to be typed in.
 *
 * Every inferred value names the refdes it came from so an engineer can
 * check it against the drawing. The four "still needed" fields are the ones
 * a schematic cannot carry: carrier (firmware), cable and shielding
 * (installation), load current (rating).
 */

const LABELS: Record<string, string> = {
  switching_device_type: 'Switching device',
  dv_dt_v_per_us: 'Edge speed dv/dt',
  cm_choke_effectiveness: 'Common-mode choke',
  y_capacitance_f: 'Y-capacitors',
  rectifier_type: 'Rectifier',
  switching_frequency_khz: 'Carrier frequency',
  cable_length_m: 'Motor cable length',
  shielding_quality: 'Cable shielding',
  load_current_a: 'Load current',
}

function formatInferred(key: string, value: number | string): string {
  if (typeof value === 'string') return value.replace(/_/g, ' ')
  if (key === 'cm_choke_effectiveness') return `${(value * 2).toFixed(2)} mH`
  if (key === 'y_capacitance_f') return `${(value * 1e9).toFixed(1)} nF`
  if (key === 'dv_dt_v_per_us') return `${value.toLocaleString()} V/µs (bin)`
  return String(value)
}

function confidenceWord(c: number): string {
  if (c >= 0.7) return 'good'
  if (c >= 0.5) return 'fair'
  return 'weak'
}

export function SchematicFindings({
  report,
  specs,
}: {
  report: SchematicImportResult
  specs: ParameterSpec[]
}) {
  const [showParts, setShowParts] = useState(false)
  const specLabel = (key: string) =>
    specs.find((s) => s.key === key)?.label ?? LABELS[key] ?? key

  return (
    <Card className="space-y-5 border-dashed">
      <div>
        <Eyebrow>Read from schematic</Eyebrow>
        <h2 className="mt-1 text-base font-semibold tracking-tight text-ink">
          {report.title ?? report.filename ?? 'Uploaded schematic'}
        </h2>
        <p className="mt-1 text-xs text-ink-faint">
          {report.filename} · {report.pages} pages · {report.part_count ?? report.parts.length} parts read
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div>
          <p className="text-2xs font-semibold uppercase tracking-label text-ink-muted">
            Pre-filled from the drawing
          </p>
          <dl className="mt-2 divide-y divide-line">
            {Object.entries(report.inferred).map(([key, row]) => (
              <div key={key} className="py-2">
                <div className="flex items-baseline justify-between gap-3">
                  <dt className="text-sm text-ink">{specLabel(key)}</dt>
                  <dd className="text-sm tabular text-ink">{formatInferred(key, row.value)}</dd>
                </div>
                <p className="mt-0.5 text-2xs text-ink-faint">
                  from {row.source} · confidence {confidenceWord(row.confidence)}
                </p>
              </div>
            ))}
          </dl>
        </div>

        <div>
          <p className="text-2xs font-semibold uppercase tracking-label text-ink-muted">
            Still needed from you
          </p>
          <ul className="mt-2 divide-y divide-line">
            {report.missing.map((key) => (
              <li key={key} className="py-2 text-sm text-ink">
                {specLabel(key)}
                <span className="ml-2 text-2xs text-ink-faint">
                  {key === 'switching_frequency_khz'
                    ? 'set in firmware, not on the drawing'
                    : key === 'load_current_a'
                      ? 'a rating, not a component'
                      : 'an installation fact'}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-2xs leading-relaxed text-ink-faint">{report.note}</p>
        </div>
      </div>

      {report.findings.length ? (
        <div>
          <p className="text-2xs font-semibold uppercase tracking-label text-ink-muted">
            EMC hardware found
          </p>
          <ul className="mt-2 space-y-2.5">
            {report.findings.map((f) => (
              <li key={f.key} className="rounded-lg border border-line bg-surface p-3">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-sm font-medium text-ink">{f.label}</span>
                  <span className="text-2xs tabular text-ink-faint">{f.detail}</span>
                </div>
                <p className="mt-1 text-xs leading-relaxed text-ink-muted">{f.emc_note}</p>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <button
        type="button"
        onClick={() => setShowParts((s) => !s)}
        className="text-xs text-ink-muted underline decoration-line-strong underline-offset-2 hover:text-ink"
      >
        {showParts ? 'Hide' : 'Show'} bill of materials ({report.parts.length})
      </button>
      {showParts ? (
        <div className="max-h-64 overflow-auto rounded-lg border border-line">
          <table className="w-full text-left text-2xs tabular">
            <thead className="sticky top-0 bg-surface text-ink-faint">
              <tr>
                <th className="px-2 py-1 font-medium">Ref</th>
                <th className="px-2 py-1 font-medium">Value</th>
                <th className="px-2 py-1 font-medium">Page</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line text-ink-muted">
              {report.parts.map((p) => (
                <tr key={p.refdes}>
                  <td className="px-2 py-1 text-ink">{p.refdes}</td>
                  <td className="px-2 py-1">{p.value || '—'}</td>
                  <td className="px-2 py-1">{p.page}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </Card>
  )
}
