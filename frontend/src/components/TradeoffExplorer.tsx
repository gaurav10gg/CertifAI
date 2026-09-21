import { useMemo } from 'react'
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { TradeoffPoint, TradeoffResponse } from '../api/types'
import { Card, Eyebrow } from './Card'
import { SpinnerIcon } from './Icons'

/**
 * TradeoffExplorer.tsx -- Design Trade-offs: EMC headroom vs the two costs
 * named by the carrier countermeasure (audible noise and motor current ripple).
 *
 * Monochrome: Pareto points are filled ink, sized by switching frequency;
 * dominated points are smaller hollow marks in faint ink. The current design
 * is a ring. The frontier is an ink polyline through Pareto points ordered
 * by EMC score. Pass/fail colour is not used — these axes are not verdicts.
 */

const F_SW_MIN = 3
const F_SW_MAX = 16

const FINDING =
  'Acoustic risk and motor current ripple are both driven by the same switching-frequency trade-off, so they highlight the same set of Pareto-optimal points here — lowering carrier frequency to improve EMC margin consistently costs both quieter operation and smoother current, together.'

type CostKey = 'acoustic_risk' | 'ripple_cost'
type ParetoKey = 'pareto_acoustic' | 'pareto_ripple'

type ChartRow = TradeoffPoint & { x: number; y: number; pareto: boolean }

function frequencySize(fKhz: number, pareto: boolean): number {
  const t = Math.min(1, Math.max(0, (fKhz - F_SW_MIN) / (F_SW_MAX - F_SW_MIN)))
  const radius = 3.6 + t * 4.2
  return pareto ? radius : radius * 0.62
}

function ChartTooltip({
  active,
  payload,
  yLabel,
}: {
  active?: boolean
  payload?: { payload: ChartRow }[]
  yLabel: string
}) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  return (
    <div className="rounded-lg border border-line bg-raised px-3 py-2 text-xs shadow-lifted">
      <p className="tabular font-medium text-ink">
        {point.switching_frequency_khz.toFixed(1)} kHz
        {point.pareto ? (
          <span className="ml-2 font-normal text-ink-faint">Pareto</span>
        ) : null}
      </p>
      <dl className="mt-1.5 space-y-0.5 tabular text-ink-muted">
        <div className="flex justify-between gap-4">
          <dt>EMC risk score</dt>
          <dd className="text-ink">{point.emc_risk_score.toFixed(1)}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt>{yLabel}</dt>
          <dd className="text-ink">{point.y.toFixed(1)}</dd>
        </div>
      </dl>
      <p className="mt-1.5 text-2xs text-ink-faint">Click to re-assess at this carrier</p>
    </div>
  )
}

function TradeoffScatter({
  points,
  yKey,
  paretoKey,
  yLabel,
  yCaption,
  currentFrequency,
  busy,
  onSelectFrequency,
}: {
  points: TradeoffPoint[]
  yKey: CostKey
  paretoKey: ParetoKey
  yLabel: string
  yCaption: string
  currentFrequency: number
  busy: boolean
  onSelectFrequency: (khz: number) => void
}) {
  const rows = useMemo<ChartRow[]>(
    () =>
      points.map((point) => ({
        ...point,
        x: point.emc_risk_score,
        y: point[yKey],
        pareto: point[paretoKey],
      })),
    [points, yKey, paretoKey],
  )

  const dominated = useMemo(() => rows.filter((row) => !row.pareto), [rows])
  const pareto = useMemo(() => rows.filter((row) => row.pareto), [rows])
  const frontier = useMemo(
    () =>
      [...pareto].sort(
        (a, b) => a.x - b.x || a.y - b.y,
      ),
    [pareto],
  )

  const pick = (row: ChartRow | undefined) => {
    if (!row || busy) return
    onSelectFrequency(row.switching_frequency_khz)
  }

  const renderDot =
    (kind: 'pareto' | 'dominated') =>
    // Recharts injects cx/cy; the rest of the props bag is unused.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (props: any) => {
      const { cx, cy, payload } = props as {
        cx?: number
        cy?: number
        payload: ChartRow
      }
      if (cx == null || cy == null) return null
      const current =
        Math.abs(payload.switching_frequency_khz - currentFrequency) < 0.05
      const radius = frequencySize(payload.switching_frequency_khz, kind === 'pareto')
      const isPareto = kind === 'pareto'
      return (
        <g
          className={busy ? 'cursor-wait' : 'cursor-pointer'}
          onClick={() => pick(payload)}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              pick(payload)
            }
          }}
          aria-label={`${payload.switching_frequency_khz.toFixed(1)} kHz, EMC score ${payload.emc_risk_score.toFixed(1)}`}
        >
          {current ? (
            <circle
              cx={cx}
              cy={cy}
              r={radius + 4}
              fill="none"
              stroke="rgb(var(--ink))"
              strokeWidth={1.4}
            />
          ) : null}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill={isPareto ? 'rgb(var(--ink))' : 'rgb(var(--raised))'}
            stroke={isPareto ? 'rgb(var(--ink))' : 'rgb(var(--ink-faint))'}
            strokeWidth={isPareto ? 0 : 1.2}
          />
        </g>
      )
    }

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-ink">{yCaption}</p>
          <p className="mt-0.5 text-xs text-ink-muted">
            Higher EMC score is more headroom. Lower {yLabel.toLowerCase()} is better.
          </p>
        </div>
      </div>

      <div className="mt-3 h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={rows}
            margin={{ top: 12, right: 12, bottom: 28, left: 8 }}
          >
            <CartesianGrid
              stroke="rgb(var(--chart-grid))"
              strokeWidth={1}
              vertical={false}
            />
            <XAxis
              type="number"
              dataKey="x"
              domain={['auto', 'auto']}
              tick={{ fill: 'rgb(var(--ink-faint))', fontSize: 11 }}
              stroke="rgb(var(--line))"
              tickLine={false}
              label={{
                value: 'EMC risk score (higher = more headroom)',
                position: 'insideBottom',
                offset: -16,
                fill: 'rgb(var(--ink-faint))',
                fontSize: 11,
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={['auto', 'auto']}
              tick={{ fill: 'rgb(var(--ink-faint))', fontSize: 11 }}
              stroke="rgb(var(--line))"
              tickLine={false}
              width={42}
              label={{
                value: yLabel,
                angle: -90,
                position: 'insideLeft',
                offset: 18,
                fill: 'rgb(var(--ink-faint))',
                fontSize: 11,
              }}
            />
            <Line
              data={frontier}
              type="linear"
              dataKey="y"
              stroke="rgb(var(--ink))"
              strokeWidth={1.4}
              dot={false}
              isAnimationActive={false}
              legendType="none"
            />
            <Scatter
              data={dominated}
              dataKey="y"
              isAnimationActive={false}
              shape={renderDot('dominated')}
            />
            <Scatter
              data={pareto}
              dataKey="y"
              isAnimationActive={false}
              shape={renderDot('pareto')}
            />
            <RechartsTooltip
              content={<ChartTooltip yLabel={yLabel} />}
              cursor={{ stroke: 'rgb(var(--line-strong))', strokeWidth: 1 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function TradeoffExplorer({
  data,
  loading,
  error,
  currentFrequency,
  applying,
  applyingFrequency,
  onSelectFrequency,
}: {
  data: TradeoffResponse | null
  loading: boolean
  error: string | null
  currentFrequency: number
  applying: boolean
  applyingFrequency: number | null
  onSelectFrequency: (khz: number) => void
}) {
  const ringFrequency = applyingFrequency ?? currentFrequency

  return (
    <Card className="sm:p-7">
      <Eyebrow>Design trade-offs</Eyebrow>
      <h2 className="mt-1 text-lg font-semibold tracking-tight text-ink">
        Switching frequency vs competing costs
      </h2>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-ink-muted">
        Each point is one carrier frequency with every other parameter held
        fixed. Filled marks are Pareto-optimal; hollow marks are dominated.
        Marker size tracks switching frequency (small = 3 kHz, large = 16 kHz).
        The ring is the configuration currently assessed. Click any point to
        load that carrier and re-run the full prediction.
      </p>

      {applying ? (
        <p className="mt-3 flex items-center gap-2 text-sm text-ink">
          <SpinnerIcon className="h-4 w-4" />
          Re-assessing at {applyingFrequency?.toFixed(1)} kHz through the full
          prediction pipeline…
        </p>
      ) : null}

      {error && data ? <p className="mt-3 text-sm text-fail">{error}</p> : null}

      {loading && !data ? (
        <div className="mt-6 flex items-center gap-3 rounded-lg border border-line bg-surface px-4 py-6 text-sm text-ink-muted">
          <SpinnerIcon className="h-4 w-4" />
          Sweeping switching frequency. This re-simulates each carrier and takes
          about a minute the first time.
        </div>
      ) : error && !data ? (
        <p className="mt-6 text-sm text-fail">{error}</p>
      ) : data ? (
        <div className={applying ? 'pointer-events-none opacity-70' : ''}>
          <div className="mt-6">
            <TradeoffScatter
              points={data.points}
              yKey="acoustic_risk"
              paretoKey="pareto_acoustic"
              yLabel="Acoustic risk"
              yCaption="EMC headroom vs acoustic risk"
              currentFrequency={ringFrequency}
              busy={applying}
              onSelectFrequency={onSelectFrequency}
            />
          </div>

          <p className="mt-6 rounded-lg border border-line border-l-2 border-l-ink bg-surface px-4 py-3 text-sm leading-relaxed text-ink">
            {FINDING}
          </p>

          <div className="mt-6">
            <TradeoffScatter
              points={data.points}
              yKey="ripple_cost"
              paretoKey="pareto_ripple"
              yLabel="Ripple cost"
              yCaption="EMC headroom vs motor current ripple"
              currentFrequency={ringFrequency}
              busy={applying}
              onSelectFrequency={onSelectFrequency}
            />
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line pt-3 text-2xs text-ink-muted">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-ink" />
              Pareto-optimal
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full border border-ink-faint bg-raised" />
              Dominated
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3.5 w-3.5 rounded-full border border-ink" />
              Current design
            </span>
            <span>Size = switching frequency</span>
          </div>

          <p className="mt-3 text-xs leading-relaxed text-ink-faint">
            {data.caption}
          </p>
        </div>
      ) : null}
    </Card>
  )
}
