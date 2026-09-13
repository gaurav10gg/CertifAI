"""End-to-end sanity check against a running API.

Sweeps one parameter through the live /api/predict endpoint and prints the
resulting scores, which is the quickest way to confirm the whole stack -- request
validation, simulation, feature extraction, both models and the scoring blend --
still responds to design changes in the physically correct direction.

Run the backend first, then:  python tools/api_check.py
"""

from __future__ import annotations

import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

# The high-speed ultra-rise preset, which fails on cable length.
BASELINE = {
    "switching_frequency_khz": 12.0,
    "dv_dt_v_per_us": 6000.0,
    "cable_length_m": 85.0,
    "shielding_quality": 0.91,
    "load_current_a": 140.0,
    "pwm_modulation_type": "SVPWM",
}


def predict(parameters: dict) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}/api/predict",
        data=json.dumps({"parameters": parameters}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def sweep(key: str, values: tuple[float, ...]) -> None:
    print(f"\n{key}")
    print(f"{'value':>10}  {'score':>5}  verdict  conf  margins (dB, band A/B/C)")
    for value in values:
        result = predict({**BASELINE, key: value})
        margins = "  ".join(
            f"{band['predicted_margin_db']:+6.1f}" for band in result["bands"]
        )
        print(
            f"{value:>10.2f}  {result['compliance_score']:5.1f}  "
            f"{result['verdict']:<7}  {result['confidence_score']:4.0f}  {margins}"
        )


def main() -> None:
    with urllib.request.urlopen(f"{BASE_URL}/api/health") as response:
        print("health:", json.load(response))

    sweep("cable_length_m", (120.0, 85.0, 40.0, 20.0, 10.0, 5.0))
    sweep("shielding_quality", (0.0, 0.3, 0.6, 0.91, 1.0))
    sweep("dv_dt_v_per_us", (500.0, 2000.0, 6000.0, 10000.0))
    sweep("load_current_a", (5.0, 40.0, 140.0, 200.0))
    sweep("switching_frequency_khz", (2.0, 8.0, 12.0, 20.0))


if __name__ == "__main__":
    main()
