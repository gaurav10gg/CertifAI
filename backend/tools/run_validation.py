"""Run the Phase 7 validation suite and write models/validation_suite.json.

From backend/:

    python tools/run_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation import run_suite, write_suite


def main() -> int:
    report = run_suite()
    path = write_suite(report)
    status = "PASS" if report["passed"] else "FAIL"
    print(f"{status}  {report['headlines']['monotonicity']}")
    print(f"{status}  {report['headlines']['shap_additivity']}")
    print(f"wrote {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
