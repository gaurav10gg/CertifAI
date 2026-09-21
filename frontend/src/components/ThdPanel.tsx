import type { PowerQualityResult } from '../api/types'
import { Eyebrow } from './Card'

/**
 * ThdPanel.tsx -- motor and input current THD, kept separate from EMC bands.
 */

function formatHz(hz: number): string {
  if (hz >= 1000) return `${(hz / 1000).toFixed(2)} kHz`
  return `${hz.toFixed(0)} Hz`
}

export function ThdPanel({ reports }: { reports: PowerQualityResult[] }) {
  if (!reports.length) return null

  return (
    <div>
      <Eyebrow>Power quality · THD</Eyebrow>
      <p className="mt-1 mb-4 text-sm text-ink-muted">
        Total harmonic distortion of the current waveforms. This is a power-
        quality metric, not a conducted-emission verdict — the two answer
        different questions.
      </p>
      <div className="grid gap-4 sm:grid-cols-2">
        {reports.map((report) => (
          <div key={report.key} className="rounded-lg border border-line bg-surface p-4">
            <p className="text-xs uppercase tracking-label text-ink-faint">
              {report.label}
            </p>
            <p className="mt-1 text-3xl font-semibold tabular tracking-tight text-ink">
              {report.thd_percent.toFixed(1)}
              <span className="ml-1 text-base font-normal text-ink-faint">%</span>
            </p>
            <p className="mt-2 text-xs leading-relaxed text-ink-muted">
              {report.dominant_statement}
            </p>
            <table className="mt-3 w-full text-left text-2xs">
              <thead>
                <tr className="border-b border-line text-ink-faint">
                  <th className="py-1 font-medium">Order</th>
                  <th className="py-1 font-medium">Freq</th>
                  <th className="py-1 text-right font-medium">% of fund.</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {report.peaks.slice(0, 5).map((peak) => (
                  <tr key={`${report.key}-${peak.order}`}>
                    <td className="py-1 tabular text-ink">{peak.order}</td>
                    <td className="py-1 tabular text-ink-muted">
                      {formatHz(peak.frequency_hz)}
                    </td>
                    <td className="py-1 text-right tabular text-ink">
                      {peak.percent_of_fundamental.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  )
}
