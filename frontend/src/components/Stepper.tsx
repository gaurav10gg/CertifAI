import { CheckIcon } from './Icons'

/**
 * Stepper.tsx -- the three-stage progress indicator.
 *
 * Completed steps are clickable so a user can go back and change an input without
 * losing their place. Current position is carried by weight and a filled marker
 * rather than colour.
 */

export type StepId = 'select' | 'configure' | 'results'

const STEPS: { id: StepId; label: string }[] = [
  { id: 'select', label: 'Select device' },
  { id: 'configure', label: 'Set parameters' },
  { id: 'results', label: 'Review results' },
]

export function Stepper({
  current,
  furthest,
  onNavigate,
}: {
  current: StepId
  /** The furthest step reached, which bounds where the user may jump back to. */
  furthest: StepId
  onNavigate: (step: StepId) => void
}) {
  const currentIndex = STEPS.findIndex((step) => step.id === current)
  const furthestIndex = STEPS.findIndex((step) => step.id === furthest)

  return (
    <nav aria-label="Progress">
      <ol className="flex items-center gap-1.5 sm:gap-3">
        {STEPS.map((step, index) => {
          const isCurrent = index === currentIndex
          const isComplete = index < currentIndex
          const reachable = index <= furthestIndex && !isCurrent

          return (
            <li key={step.id} className="flex items-center gap-1.5 sm:gap-3">
              <button
                type="button"
                onClick={reachable ? () => onNavigate(step.id) : undefined}
                disabled={!reachable}
                aria-current={isCurrent ? 'step' : undefined}
                className={`flex items-center gap-2 rounded px-1 py-0.5 transition-colors
                  ${reachable ? 'hover:text-ink' : ''}
                  ${isCurrent ? 'text-ink' : 'text-ink-faint'}
                  ${reachable ? 'cursor-pointer' : 'cursor-default'}`}
              >
                <span
                  className={`flex h-[18px] w-[18px] shrink-0 items-center justify-center
                    rounded-full border text-2xs font-semibold tabular
                    ${
                      isComplete
                        ? 'border-accent bg-accent text-accent-ink'
                        : isCurrent
                          ? 'border-accent text-ink'
                          : 'border-line text-ink-faint'
                    }`}
                >
                  {isComplete ? <CheckIcon className="h-2.5 w-2.5" /> : index + 1}
                </span>
                <span
                  className={`hidden text-xs sm:inline ${
                    isCurrent ? 'font-medium' : ''
                  }`}
                >
                  {step.label}
                </span>
              </button>

              {index < STEPS.length - 1 ? (
                <span
                  aria-hidden="true"
                  className={`h-px w-6 sm:w-10 ${
                    index < currentIndex ? 'bg-line-strong' : 'bg-line'
                  }`}
                />
              ) : null}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
