"""Dump the Standard Gearless /tradeoff sweep as JSON for inspection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from devices import DEVICE_PROFILE_BY_ID  # noqa: E402
from tradeoff import run_tradeoff  # noqa: E402


def main() -> None:
    profile = DEVICE_PROFILE_BY_ID["standard-gearless-vfd"]
    result = run_tradeoff(profile.parameters, n_points=27)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
