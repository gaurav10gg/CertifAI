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

import type { ShapRow } from '../api/types'
import { Eyebrow } from './Card'

/**
 * ShapWaterfall.tsx -- exact tree SHAP contributions for the worst band.
 *
 * Positive shap raises predicted margin (green); negative hurts it (red).
 * Rows are ordered by |contribution|, largest magnitude at the top.
 */

type BarPoint = ShapRow & { magnitude: number }

function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: { payload: BarPoint }[]
}) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload
  return (
    <div className="rounded-lg border border-line bg-raised px-3 py-2 text-xs shadow-lifted">
      <p className="font-medium text-ink">{row.label}</p>
      <p className="mt-1 tabular text-ink-muted">
        {row.shap >= 0 ? '+' : ''}
        {row.shap.toFixed(2)} dB
        {row.raises_risk ? ' · reduces margin' : ' · increases margin'}
      </p>
    </div>
  )
}

export function ShapWaterfall({
  contributions,
  band,
}: {
  contributions: ShapRow[]
  band: string
  note?: string
}) {
  const data = useMemo<BarPoint[]>(
    () =>
      [...contributions]
        .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
        .map((row) => ({ ...row, magnitude: Math.abs(row.shap) })),
    [contributions],
  )

  if (!data.length) return null

  return (
    <div>
      <Eyebrow>Why this margin · {band}</Eyebrow>
      <p className="mt-1 mb-4 text-sm text-ink-muted">
        Exact attribution for this specific configuration.
      </p>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 12, left: 8, bottom: 18 }}
          >
            <CartesianGrid stroke="rgb(var(--line))" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: 'rgb(var(--ink-faint))', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              label={{
                value: 'Contribution to predicted margin (dB)',
                position: 'insideBottom',
                offset: -2,
                fill: 'rgb(var(--ink-faint))',
                fontSize: 11,
              }}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={118}
              reversed
              tick={{ fill: 'rgb(var(--ink-muted))', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <RechartsTooltip content={<ChartTooltip />} cursor={{ fill: 'rgb(var(--surface))' }} />
            <Bar dataKey="shap" radius={[0, 3, 3, 0]} maxBarSize={14}>
              {data.map((row) => (
                <Cell
                  key={row.parameter}
                  fill={
                    row.raises_risk
                      ? 'rgb(var(--fail))'
                      : 'rgb(var(--pass))'
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
