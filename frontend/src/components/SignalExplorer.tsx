import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { SignalTrace } from '../api/types'
import { Eyebrow } from './Card'

/**
 * SignalExplorer.tsx -- tabbed time-domain viewer for the five simulated
 * waveforms (DC-link, motor voltage, motor current, common-mode, input current).
 */

type ChartPoint = { t: number; y: number }

function ChartTooltip({
  active,
  payload,
  unit,
}: {
  active?: boolean
  payload?: { payload: ChartPoint }[]
  unit: string
}) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  return (
    <div className="rounded-lg border border-line bg-raised px-3 py-2 text-xs shadow-lifted">
      <p className="tabular text-ink-muted">{point.t.toFixed(3)} ms</p>
      <p className="mt-0.5 tabular font-medium text-ink">
        {point.y.toFixed(2)} {unit}
      </p>
    </div>
  )
}

export function SignalExplorer({ signals }: { signals: SignalTrace[] }) {
  const [activeKey, setActiveKey] = useState(signals[0]?.key ?? '')
  const active = signals.find((s) => s.key === activeKey) ?? signals[0]

  const data = useMemo<ChartPoint[]>(() => {
    if (!active) return []
    return active.time_ms.map((t, i) => ({ t, y: active.values[i] ?? 0 }))
  }, [active])

  if (!signals.length || !active) {
    return (
      <p className="text-sm text-ink-muted">No time-domain traces were returned.</p>
    )
  }

  return (
    <div>
      <Eyebrow>Signal explorer</Eyebrow>
      <p className="mt-1 mb-4 text-sm text-ink-muted">
        Five distinct waveforms for this configuration. Common-mode is the
        conducted-EMC driver; the currents are for power-quality THD, not the
        emission limit.
      </p>

      <div className="flex flex-wrap gap-1.5">
        {signals.map((signal) => {
          const selected = signal.key === active.key
          return (
            <button
              key={signal.key}
              type="button"
              onClick={() => setActiveKey(signal.key)}
              className={`rounded-md border px-2.5 py-1 text-2xs font-medium transition-colors
                ${
                  selected
                    ? 'border-accent bg-accent text-accent-ink'
                    : 'border-line bg-transparent text-ink-muted hover:border-line-strong hover:text-ink'
                }`}
            >
              {signal.label}
            </button>
          )
        })}
      </div>

      <p className="mt-3 text-xs leading-relaxed text-ink-muted">
        {active.description}
      </p>

      <div className="mt-4 h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="rgb(var(--line))" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              tick={{ fill: 'rgb(var(--ink-faint))', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: 'rgb(var(--line))' }}
              tickFormatter={(value: number) => `${value.toFixed(2)}`}
              label={{
                value: 'ms',
                position: 'insideBottomRight',
                offset: -2,
                fill: 'rgb(var(--ink-faint))',
                fontSize: 11,
              }}
            />
            <YAxis
              tick={{ fill: 'rgb(var(--ink-faint))', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={48}
              tickFormatter={(value: number) => `${value.toFixed(0)}`}
            />
            <RechartsTooltip
              content={<ChartTooltip unit={active.unit} />}
              cursor={{ stroke: 'rgb(var(--line-strong))' }}
            />
            <Line
              type="monotone"
              dataKey="y"
              stroke="rgb(var(--ink))"
              strokeWidth={1.4}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
