import type { DeviceProfile } from '../api/types'
import { Eyebrow, SelectableCard } from '../components/Card'
import { ArrowRightIcon, CustomIcon, DeviceIcon } from '../components/Icons'

/**
 * DeviceSelect.tsx -- stage one: pick a starting point.
 *
 * The presets are starting points in the design space, not shortcuts to a
 * certification result. Two of them are deliberately high-risk configurations,
 * so the tool can demonstrate a mitigation path without the user inventing one.
 */

export function DeviceSelect({
  devices,
  selectedId,
  onSelect,
  onCustom,
  loading,
}: {
  devices: DeviceProfile[]
  selectedId: string | null
  onSelect: (device: DeviceProfile) => void
  onCustom: () => void
  loading: boolean
}) {
  return (
    <div className="motion-safe:animate-fade-up">
      <header className="max-w-2xl">
        <Eyebrow>Step 1</Eyebrow>
        <h1 className="mt-2 text-display font-semibold tracking-tight text-ink">
          Select a drive configuration
        </h1>
        <p className="mt-3 text-base leading-relaxed text-ink-muted">
          Start from a representative elevator drive profile, then adjust its
          parameters. The assessment that follows is a pre-compliance risk
          indicator to guide design decisions — not a certification prediction.
        </p>
      </header>

      {loading ? (
        <DeviceGridSkeleton />
      ) : (
        <div className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {devices.map((device) => (
            <SelectableCard
              key={device.id}
              selected={device.id === selectedId}
              onClick={() => onSelect(device)}
              ariaLabel={`Select ${device.name}`}
            >
              <div className="flex w-full items-start justify-between gap-3">
                <DeviceIcon
                  name={device.icon}
                  className="h-6 w-6 shrink-0 text-ink"
                />
                <span className="label-eyebrow">{device.category}</span>
              </div>

              <h2 className="mt-4 text-base font-semibold leading-snug tracking-tight text-ink">
                {device.name}
              </h2>
              <p className="mt-1.5 text-sm leading-relaxed text-ink-muted">
                {device.summary}
              </p>

              <ul className="mt-4 w-full space-y-1 border-t border-line pt-3">
                {device.highlights.map((highlight) => (
                  <li
                    key={highlight}
                    className="flex gap-2 text-2xs tabular text-ink-faint"
                  >
                    <span aria-hidden="true" className="select-none">
                      ·
                    </span>
                    {highlight}
                  </li>
                ))}
              </ul>
            </SelectableCard>
          ))}

          {/* Custom configuration: same grid, visually distinguished by an
              outline-only treatment so it reads as an alternative, not a preset. */}
          <button
            type="button"
            onClick={onCustom}
            className="group flex min-h-[200px] flex-col items-start justify-center
              rounded-card border border-dashed border-line-strong bg-transparent p-5
              text-left transition-all duration-200 ease-standard
              hover:border-accent hover:bg-surface"
          >
            <CustomIcon className="h-6 w-6 text-ink-muted transition-colors group-hover:text-ink" />
            <h2 className="mt-4 text-base font-semibold tracking-tight text-ink">
              Custom configuration
            </h2>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-muted">
              Enter every parameter yourself, starting from mid-range defaults.
            </p>
            <span className="mt-4 inline-flex items-center gap-1.5 text-xs font-medium text-ink">
              Configure manually
              <ArrowRightIcon className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
            </span>
          </button>
        </div>
      )}
    </div>
  )
}

function DeviceGridSkeleton() {
  return (
    <div
      className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
      aria-hidden="true"
    >
      {Array.from({ length: 6 }).map((_, index) => (
        <div
          key={index}
          className="min-h-[212px] animate-pulse rounded-card border border-line bg-surface"
        />
      ))}
    </div>
  )
}
