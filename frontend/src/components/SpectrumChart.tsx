import { useMemo } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { BandResult, SpectrumTrace } from '../api/types'
import { Eyebrow } from './Card'

/**
 * SpectrumChart.tsx -- simulated emission trace against the assumed limit line.
 *
 * Styling rules, consistent with the rest of the product:
 *   * Both curves are drawn in ink. They are distinguished by line style (solid
 *     emission, dashed limit), not by colour.
 *   * The region above the limit is shaded, so an exceedance is visible as
 *     geometry rather than needing a legend lookup.
 *   * Where the emission crosses into that region, the excess is filled with the
 *     fail colour at low opacity. This is the one place colour appears, and it
 *     duplicates information the shading already conveys.
 *   * The x axis is logarithmic, which is how conducted-emission spectra are
 *     always presented.
 */

type ChartPoint = {
  frequencyMhz: number
  emission: number
  limit: number
  /**
   * Height from the limit line to the top of the plot. Stacked on top of `limit`
   * with a shared stackId, this is how the non-compliant region gets filled from
   * the limit line upwards -- Recharts has no native band-fill between two
   * series, and an unstacked area would fill from the axis base instead.
   */
  zoneHeight: number
  /**
   * Emission level only where it exceeds the limit, null elsewhere. Drawn as a
   * separate line with connectNulls disabled, so the emission curve turns red
   * over exactly the frequencies that are out of compliance.
   */
  exceedance: number | null
}

function formatFrequency(mhz: number): string {
  if (mhz < 1) return `${Math.round(mhz * 1000)}k`
  return `${mhz % 1 === 0 ? mhz : mhz.toFixed(1)}M`
}

function ChartTooltip({
  active,
  payload,
  unit = 'dBµV',
}: {
  active?: boolean
  payload?: { payload: ChartPoint }[]
  unit?: string
}) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  const margin = point.limit - point.emission
  const over = margin < 0

  return (
    <div className="rounded-lg border border-line bg-raised px-3 py-2 text-xs shadow-lifted">
      <p className="tabular font-medium text-ink">
        {point.frequencyMhz < 1
          ? `${(point.frequencyMhz * 1000).toFixed(0)} kHz`
          : `${point.frequencyMhz.toFixed(2)} MHz`}
      </p>
      <dl className="mt-1.5 space-y-0.5 tabular text-ink-muted">
        <div className="flex justify-between gap-4">
          <dt>Emission</dt>
          <dd className="text-ink">{point.emission.toFixed(1)} {unit}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt>Limit</dt>
          <dd className="text-ink">{point.limit.toFixed(1)} {unit}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt>Margin</dt>
          <dd className={over ? 'text-fail' : 'text-pass'}>
            {margin >= 0 ? '+' : ''}
            {margin.toFixed(1)} dB
          </dd>
        </div>
      </dl>
    </div>
  )
}

export function SpectrumChart({
  spectrum,
  bands,
  eyebrow = 'Emission spectrum',
  subtitle = 'Simulated max-hold trace against the assumed limit line, 9 kHz resolution bandwidth.',
  unit = 'dBµV',
  yLabel = 'dBµV',
  xTicks = [0.15, 0.3, 0.5, 1, 2, 5, 10, 20, 30],
  dividersMhz = [0.5, 5],
  chartId = 'conducted',
  showDots = false,
}: {
  spectrum: SpectrumTrace
  bands: BandResult[]
  eyebrow?: string
  subtitle?: string
  unit?: string
  yLabel?: string
  xTicks?: number[]
  dividersMhz?: number[]
  chartId?: string
  showDots?: boolean
}) {
  const { yMin, yMax } = useMemo(() => {
    const values = [...spectrum.emission_dbuv, ...spectrum.limit_dbuv]
    return {
      yMin: Math.floor((Math.min(...values) - 6) / 10) * 10,
      yMax: Math.ceil((Math.max(...values) + 8) / 10) * 10,
    }
  }, [spectrum])

  const data = useMemo<ChartPoint[]>(
    () =>
      spectrum.frequency_hz.map((hz, index) => {
        const emission = spectrum.emission_dbuv[index]
        const limit = spectrum.limit_dbuv[index]
        return {
          frequencyMhz: hz / 1e6,
          emission,
          limit,
          zoneHeight: Math.max(yMax - limit, 0),
          exceedance: emission > limit ? emission : null,
        }
      }),
    [spectrum, yMax],
  )

  const hasExceedance = data.some((point) => point.exceedance !== null)

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Eyebrow>{eyebrow}</Eyebrow>
          <p className="mt-1 text-sm text-ink-muted">{subtitle}</p>
        </div>

        <div className="flex items-center gap-4 text-xs text-ink-muted">
          <span className="flex items-center gap-1.5">
            <svg width="22" height="8" aria-hidden="true">
              <line
                x1="0"
                y1="4"
                x2="22"
                y2="4"
                stroke="rgb(var(--ink))"
                strokeWidth="1.5"
              />
            </svg>
            Emission
          </span>
          <span className="flex items-center gap-1.5">
            <svg width="22" height="8" aria-hidden="true">
              <line
                x1="0"
                y1="4"
                x2="22"
                y2="4"
                stroke="rgb(var(--ink))"
                strokeWidth="1.5"
                strokeDasharray="5 3"
              />
            </svg>
            Limit
          </span>
        </div>
      </div>

      <div className="mt-5 h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 8, right: 8, bottom: 22, left: 4 }}
          >
            <defs>
              {/* Non-compliant region: ink wash above the limit line. */}
              <linearGradient id={`${chartId}-limitZone`} x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="0%"
                  stopColor="rgb(var(--chart-zone))"
                  stopOpacity="var(--chart-zone-alpha)"
                />
                <stop
                  offset="100%"
                  stopColor="rgb(var(--chart-zone))"
                  stopOpacity="var(--chart-zone-alpha)"
                />
              </linearGradient>
            </defs>

            <CartesianGrid
              stroke="rgb(var(--chart-grid))"
              strokeWidth={1}
              vertical={false}
            />

            <XAxis
              dataKey="frequencyMhz"
              type="number"
              scale="log"
              domain={['dataMin', 'dataMax']}
              ticks={xTicks}
              tickFormatter={formatFrequency}
              tick={{ fill: 'rgb(var(--ink-faint))', fontSize: 11 }}
              stroke="rgb(var(--line))"
              tickLine={false}
              label={{
                value: 'Frequency (Hz)',
                position: 'insideBottom',
                offset: -12,
                fill: 'rgb(var(--ink-faint))',
                fontSize: 11,
              }}
            />
            <YAxis
              domain={[yMin, yMax]}
              tick={{ fill: 'rgb(var(--ink-faint))', fontSize: 11 }}
              stroke="rgb(var(--line))"
              tickLine={false}
              width={46}
              label={{
                value: yLabel,
                angle: -90,
                position: 'insideLeft',
                offset: 14,
                fill: 'rgb(var(--ink-faint))',
                fontSize: 11,
              }}
            />

            {/* Band boundaries, as faint verticals. */}
            {dividersMhz.map((mhz) => (
              <ReferenceLine
                key={mhz}
                x={mhz}
                stroke="rgb(var(--line))"
                strokeDasharray="2 3"
              />
            ))}

            {/* Non-compliant region. The invisible `limit` area lifts the stack
                to the limit line, then `zoneHeight` washes everything above it. */}
            <Area
              type="monotone"
              dataKey="limit"
              stackId="zone"
              stroke="none"
              fill="none"
              fillOpacity={0}
              isAnimationActive={false}
              activeDot={false}
            />
            <Area
              type="monotone"
              dataKey="zoneHeight"
              stackId="zone"
              stroke="none"
              fill={`url(#${chartId}-limitZone)`}
              isAnimationActive={false}
              activeDot={false}
            />

            <Line
              type="monotone"
              dataKey="limit"
              stroke="rgb(var(--ink))"
              strokeWidth={1.5}
              strokeDasharray="5 3"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="emission"
              stroke="rgb(var(--ink))"
              strokeWidth={1.4}
              dot={showDots ? { r: 2.5, fill: 'rgb(var(--ink))' } : false}
              animationDuration={600}
            />
            {/* Redraw the emission curve in the fail colour over exactly the
                frequencies that breach the limit. */}
            {hasExceedance ? (
              <Line
                type="monotone"
                dataKey="exceedance"
                stroke="rgb(var(--fail))"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            ) : null}

            <RechartsTooltip
              content={<ChartTooltip unit={unit} />}
              cursor={{ stroke: 'rgb(var(--line-strong))', strokeWidth: 1 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Band ranges, aligned conceptually with the chart's three regions. */}
      <div
        className="mt-3 grid gap-2 border-t border-line pt-3"
        style={{ gridTemplateColumns: `repeat(${Math.max(bands.length, 1)}, minmax(0, 1fr))` }}
      >
        {bands.map((band) => (
          <div key={band.key} className="text-center">
            <p className="text-2xs uppercase tracking-label text-ink-faint">
              {band.label}
            </p>
            <p
              className={`mt-0.5 text-xs font-medium tabular ${
                band.passes ? 'text-ink-muted' : 'text-fail'
              }`}
            >
              {band.predicted_margin_db >= 0 ? '+' : ''}
              {band.predicted_margin_db.toFixed(1)} dB
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
