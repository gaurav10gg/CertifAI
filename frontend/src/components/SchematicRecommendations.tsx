import type {
  Countermeasure,
  PredictionResult,
  SchematicImportResult,
} from '../api/types'
import { Card, Eyebrow } from './Card'

/**
 * SchematicRecommendations.tsx -- the countermeasure list, tied back to the
 * parts on the uploaded drawing.
 *
 * "Fit a common-mode choke" is a generic suggestion. "L16 and L18 give you
 * 1.0 mH; the model wants about 1.3 mH" is something an engineer can act on.
 * Each row names the refdes the extractor read and says whether the change
 * is a value swap on an existing part, a new part, or a firmware setting.
 */

type Row = {
  key: string
  title: string
  current: string
  target: string
  action: string
  refdes: string[]
}

function buildRows(
  result: PredictionResult,
  report: SchematicImportResult,
): Row[] {
  const rows: Row[] = []
  const inferred = report.inferred
  const p = result.parameters
  const measures = new Map<string, Countermeasure>(
    (result.countermeasures ?? []).map((m) => [m.parameter, m]),
  )

  // Common-mode choke: existing parts or a new one.
  const chokeSrc = inferred.cm_choke_effectiveness
  const chokeNow = p.cm_choke_effectiveness * 2
  const chokeMeasure = measures.get('cm_choke_effectiveness')
  if (chokeMeasure) {
    const target = Number(chokeMeasure.suggested_value)
    const hasParts = chokeSrc && !String(chokeSrc.source).startsWith('no ')
    rows.push({
      key: 'choke',
      title: 'Common-mode choke',
      current: `${chokeNow.toFixed(2)} mH${hasParts ? ` (${chokeSrc.source})` : ' (none on drawing)'}`,
      target: `${target.toFixed(1)} mH`,
      action: hasParts
        ? `Increase the inductance of ${chokeSrc.source} or add a second choke in series.`
        : 'Add a multi-winding common-mode choke on the motor output.',
      refdes: hasParts ? String(chokeSrc.source).split(', ') : [],
    })
  }

  // Y-capacitors: name the existing part.
  const ySrc = inferred.y_capacitance_f
  const yNow = (p.y_capacitance_f ?? 0) * 1e9
  if (ySrc && yNow < 4.7) {
    const hasParts = !String(ySrc.source).startsWith('no ')
    rows.push({
      key: 'ycap',
      title: 'Y-capacitors to earth',
      current: `${yNow.toFixed(1)} nF${hasParts ? ` (${ySrc.source})` : ''}`,
      target: '4.7–22 nF',
      action: hasParts
        ? `${ySrc.source} is small. A larger value at the same earth point shunts more common-mode current. Check the installation's leakage-current limit first.`
        : 'Add Y-capacitors from the DC bus to the earth screw, close to the cable gland.',
      refdes: hasParts ? String(ySrc.source).split(', ') : [],
    })
  }

  // Gate resistors -> dv/dt.
  const dvMeasure = measures.get('dv_dt_v_per_us')
  const gateSrc = inferred.dv_dt_v_per_us
  if (dvMeasure && gateSrc) {
    rows.push({
      key: 'gate',
      title: 'Switching edge speed',
      current: `${Number(dvMeasure.current_value).toLocaleString()} V/µs (gate resistors ${gateSrc.source})`,
      target: `${Number(dvMeasure.suggested_value).toLocaleString()} V/µs`,
      action: `Raise the gate resistors named above by roughly 50 % to slow the edge, or add an output dv/dt filter. Costs switching loss.`,
      refdes: String(gateSrc.source).split(', '),
    })
  }

  // Carrier and modulation are firmware.
  const fMeasure = measures.get('switching_frequency_khz')
  if (fMeasure) {
    rows.push({
      key: 'carrier',
      title: 'Carrier frequency',
      current: `${Number(fMeasure.current_value).toFixed(1)} kHz`,
      target: `${Number(fMeasure.suggested_value).toFixed(1)} kHz`,
      action: 'Firmware setting, no hardware change. Lower carrier means more audible noise and current ripple.',
      refdes: [],
    })
  }
  const pwmMeasure = measures.get('pwm_cm_penalty_db')
  if (pwmMeasure) {
    rows.push({
      key: 'pwm',
      title: 'Modulation strategy',
      current: String(pwmMeasure.current_value),
      target: String(pwmMeasure.suggested_value),
      action: 'Firmware setting. DPWM removes about a third of commutations.',
      refdes: [],
    })
  }

  return rows
}

export function SchematicRecommendations({
  result,
  report,
}: {
  result: PredictionResult
  report: SchematicImportResult
}) {
  const rows = buildRows(result, report)
  if (!rows.length) return null

  return (
    <Card className="sm:p-7 border-dashed">
      <Eyebrow>Changes on your drawing</Eyebrow>
      <h2 className="mt-1 text-lg font-semibold tracking-tight text-ink">
        Where the recommendations land on {report.title ?? 'the schematic'}
      </h2>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-ink-muted">
        The same "what to change next" list, tied back to the parts the
        extractor read. Values are the model's targets; part numbers are yours
        to pick.
      </p>
      <ol className="mt-5 space-y-3">
        {rows.map((row, i) => (
          <li key={row.key} className="rounded-lg border border-line bg-surface p-4">
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <p className="text-sm font-medium text-ink">
                <span className="mr-2 tabular text-ink-faint">{String(i + 1).padStart(2, '0')}</span>
                {row.title}
              </p>
              <p className="text-xs tabular text-ink-muted">
                {row.current} <span className="text-ink-faint">→</span>{' '}
                <span className="text-ink">{row.target}</span>
              </p>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">{row.action}</p>
            {row.refdes.length ? (
              <p className="mt-2 flex flex-wrap gap-1.5">
                {row.refdes.map((r) => (
                  <span
                    key={r}
                    className="rounded border border-line-strong px-1.5 py-0.5 text-2xs tabular text-ink"
                  >
                    {r}
                  </span>
                ))}
              </p>
            ) : (
              <p className="mt-2 text-2xs text-ink-faint">firmware · no part change</p>
            )}
          </li>
        ))}
      </ol>
    </Card>
  )
}
