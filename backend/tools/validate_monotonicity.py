"""Sanity-check that monotone_constraints actually bind in the trained models.

Thin wrapper around :mod:`validation`. Prefer ``python tools/run_validation.py``,
which also records SHAP additivity. This script still exists so older README
commands keep working.

Run from backend/:

    python tools/validate_monotonicity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation import run_suite, write_suite


def main() -> int:
    # Full suite so a README-era command still refreshes the numbers the UI shows.
    report = run_suite()
    write_suite(report)
    mono = report["monotonicity"]
    status = "PASS" if mono["passed"] else "FAIL"
    print(
        f"{status}  {mono['n_checks'] - mono['n_violations']}/{mono['n_checks']} "
        "constrained sweeps monotonic"
    )
    print(mono["headline"])
    print(report["headlines"]["shap_additivity"])
    if not mono["passed"]:
        for row in mono.get("first_failures") or []:
            print(f"  {row['feature']}: illegal Δscore {row['max_illegal_delta']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
