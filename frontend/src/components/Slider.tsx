import { useState } from 'react'

import { Tooltip } from './Tooltip'

/**
 * Slider.tsx -- a labelled parameter control pairing a range input with a
 * directly editable numeric field.
 *
 * The number field is deliberately editable rather than display-only: an engineer
 * assessing a specific design knows the exact cable length and should not have to
 * hunt for it with a drag. The field holds its own draft string while focused so
 * typing "2" on the way to "25" does not immediately clamp to the minimum, and
 * commits on blur or Enter.
 */

type SliderProps = {
  label: string
  unit: string
  min: number
  max: number
  step: number
  value: number
  description: string
  /** Decimal places for display; inferred from `step` when omitted. */
  precision?: number
  onChange: (value: number) => void
}

export function Slider({
  label,
  unit,
  min,
  max,
  step,
  value,
  description,
  precision,
  onChange,
}: SliderProps) {
  const decimals = precision ?? (step < 1 ? String(step).split('.')[1].length : 0)
  const [draft, setDraft] = useState<string | null>(null)
  const [emitted, setEmitted] = useState<number | null>(null)

  // Discard the draft when the value changes from outside (preset selection,
  // reset) but not when it is our own edit echoing back, which would rewrite the
  // user's half-typed text. Adjusted during render rather than in an effect so
  // the field never paints a frame of stale text.
  const [lastValue, setLastValue] = useState(value)
  if (value !== lastValue) {
    setLastValue(value)
    if (value !== emitted) setDraft(null)
  }

  const emit = (next: number) => {
    setEmitted(next)
    onChange(next)
  }

  /**
   * Commit as the user types, but only while the typed number is inside the
   * modelled range: typing "1" on the way to "16" must not clamp to the minimum
   * and fight the next keystroke. Out-of-range text is held in the draft and
   * clamped when the field is left.
   */
  const handleType = (raw: string) => {
    setDraft(raw)
    const parsed = Number.parseFloat(raw)
    if (!Number.isNaN(parsed) && parsed >= min && parsed <= max) emit(parsed)
  }

  const commit = (raw: string) => {
    const parsed = Number.parseFloat(raw)
    setDraft(null)
    if (Number.isNaN(parsed)) return
    emit(Math.min(max, Math.max(min, parsed)))
  }

  const percent = ((value - min) / (max - min)) * 100

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label className="flex items-center gap-1.5 text-sm font-medium text-ink">
          {label}
          <Tooltip content={description} label={`About ${label}`} />
        </label>

        <div className="flex items-baseline gap-1.5">
          <input
            type="number"
            inputMode="decimal"
            min={min}
            max={max}
            step={step}
            value={draft ?? value.toFixed(decimals)}
            onChange={(event) => handleType(event.target.value)}
            onBlur={(event) => commit(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.currentTarget.blur()
              }
            }}
            aria-label={`${label} value`}
            className="w-24 rounded-md border border-line bg-paper px-2 py-1
              text-right text-sm tabular text-ink transition-colors
              hover:border-line-strong focus:border-accent focus:outline-none
              [appearance:textfield]
              [&::-webkit-inner-spin-button]:appearance-none
              [&::-webkit-outer-spin-button]:appearance-none"
          />
          {unit ? (
            <span className="w-8 text-xs text-ink-faint">{unit}</span>
          ) : (
            <span className="w-8" />
          )}
        </div>
      </div>

      <div className="relative mt-2">
        {/* Filled portion of the track, drawn under the native input. */}
        <div className="pointer-events-none absolute left-0 top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-accent"
          style={{ width: `${percent}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number.parseFloat(event.target.value))}
          aria-label={label}
          className="relative"
        />
      </div>

      <div className="mt-0.5 flex justify-between text-2xs tabular text-ink-faint">
        <span>
          {min.toFixed(decimals)} {unit}
        </span>
        <span>
          {max.toFixed(decimals)} {unit}
        </span>
      </div>
    </div>
  )
}

/**
 * A segmented control for the PWM strategy.
 *
 * Presented as discrete buttons rather than a <select> because there are only
 * four options and the modelled dB penalty for each is worth showing inline --
 * it makes the model's reasoning visible at the point of choosing.
 */
export function SegmentedControl<T extends string>({
  label,
  description,
  options,
  value,
  onChange,
}: {
  label: string
  description: string
  options: { value: T; label: string; hint?: string }[]
  value: T
  onChange: (value: T) => void
}) {
  return (
    <div>
      <label className="flex items-center gap-1.5 text-sm font-medium text-ink">
        {label}
        <Tooltip content={description} label={`About ${label}`} />
      </label>

      <div
        role="radiogroup"
        aria-label={label}
        className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4"
      >
        {options.map((option) => {
          const selected = option.value === value
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => onChange(option.value)}
              className={`rounded-lg border px-3 py-2.5 text-left transition-all
                duration-150 ease-standard
                ${
                  selected
                    ? 'border-accent bg-accent text-accent-ink'
                    : 'border-line bg-transparent text-ink hover:border-line-strong hover:bg-surface'
                }`}
            >
              <span className="block text-xs font-medium leading-tight">
                {option.label}
              </span>
              {option.hint ? (
                <span
                  className={`mt-1 block text-2xs tabular tracking-normal ${
                    selected ? 'text-accent-ink/65' : 'text-ink-faint'
                  }`}
                >
                  {option.hint}
                </span>
              ) : null}
            </button>
          )
        })}
      </div>
    </div>
  )
}
