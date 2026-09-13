"""Development helper: evaluate candidate device presets.

Run from the backend directory:
    python tools/tune_presets.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import predict
from simulate import DeviceParameters

CANDIDATES = {
    "standard-gearless-vfd": dict(
        switching_frequency_khz=8.0, dv_dt_v_per_us=3000.0, cable_length_m=30.0,
        shielding_quality=0.78, load_current_a=45.0, pwm_modulation_type="SPWM"),
    "high-speed-ultra-rise": dict(
        switching_frequency_khz=12.0, dv_dt_v_per_us=6000.0, cable_length_m=85.0,
        shielding_quality=0.91, load_current_a=140.0, pwm_modulation_type="SVPWM"),
    "compact-mrl-controller": dict(
        switching_frequency_khz=16.0, dv_dt_v_per_us=4500.0, cable_length_m=12.0,
        shielding_quality=0.45, load_current_a=22.0, pwm_modulation_type="SPWM"),
    "freight-heavy-duty": dict(
        switching_frequency_khz=4.0, dv_dt_v_per_us=1200.0, cable_length_m=40.0,
        shielding_quality=0.40, load_current_a=180.0, pwm_modulation_type="DPWM"),
    "sic-regenerative": dict(
        switching_frequency_khz=20.0, dv_dt_v_per_us=9500.0, cable_length_m=28.0,
        shielding_quality=0.96, load_current_a=70.0, pwm_modulation_type="RANDOM_SPWM"),
    "escalator-vvvf": dict(
        switching_frequency_khz=6.0, dv_dt_v_per_us=2200.0, cable_length_m=18.0,
        shielding_quality=0.70, load_current_a=75.0, pwm_modulation_type="SVPWM"),
}


def main() -> None:
    print(f"{'preset':26s} {'verdict':>7s} {'score':>6s} {'conf':>6s}  margins (A/B/C dB)")
    for key, kwargs in CANDIDATES.items():
        result = predict(DeviceParameters(**kwargs))
        margins = "  ".join(
            f"{b['key'][-1].upper()}{b['predicted_margin_db']:+7.1f}"
            for b in result["bands"]
        )
        print(f"{key:26s} {result['verdict']:>7s} "
              f"{result['compliance_score']:6.1f} {result['confidence_score']:6.1f}  "
              f"{margins}   risk={result['top_risk_factor']['parameter']}")


if __name__ == "__main__":
    main()
