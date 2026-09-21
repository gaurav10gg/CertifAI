import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { SensitivityBar } from '../api/types'
import { Eyebrow } from './Card'

/**
 * TornadoChart.tsx -- local sensitivity of the risk score from here.
 *
 * Positive delta_score means the modelled nudge lowered EMC risk (raised the
 * score). Sorted by |delta| so the longest lever is at the top.
 */

type BarPoint = SensitivityBar & { abs: number }

function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: { payload: BarPoint }[]
}) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload
  const sign = row.delta_score >= 0 ? '+' : ''
  return (
    <div className="rounded-lg border border-line bg-raised px-3 py-2 text-xs shadow-lifted">
      <p className="font-medium text-ink">{row.label}</p>
      <p className="mt-1 tabular text-ink-muted">
        Δ score {sign}
        {row.delta_score.toFixed(2)}
      </p>
    </div>
  )
}

export function TornadoChart({
  bars,
  note,
}: {
  bars: SensitivityBar[]
  note: string
}) {
  const data = useMemo<BarPoint[]>(
    () => bars.map((bar) => ({ ...bar, abs: Math.abs(bar.delta_score) })),
    [bars],
  )

  if (!data.length) return null

  return (
    <div>
      <Eyebrow>Sensitivity from this design</Eyebrow>
      <p className="mt-1 mb-4 text-sm text-ink-muted">{note}</p>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 12, left: 8, bottom: 4 }}
          >
            <CartesianGrid stroke="rgb(var(--line))" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: 'rgb(var(--ink-faint))', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={118}
              tick={{ fill: 'rgb(var(--ink-muted))', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <RechartsTooltip content={<ChartTooltip />} cursor={{ fill: 'rgb(var(--surface))' }} />
            <Bar dataKey="delta_score" radius={[0, 3, 3, 0]} maxBarSize={14}>
              {data.map((row) => (
                <Cell
                  key={row.parameter}
                  fill={
                    row.delta_score >= 0
                      ? 'rgb(var(--pass))'
                      : 'rgb(var(--fail))'
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
