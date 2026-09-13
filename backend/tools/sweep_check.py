"""Development helper: sanity-check simulator calibration and monotonicity.

Not part of the shipped API. Run from the backend directory:
    python tools/sweep_check.py

Single-parameter sweeps use a FIXED noise seed so the parameter's own effect is
isolated from the randomised noise floor. The random-population section uses the
default per-design seeds, i.e. the behaviour the API actually exposes.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from features import EMC_BANDS, band_margins, extract_features
from simulate import (
    DeviceParameters,
    PWM_MODULATION_TYPES,
    sample_random_parameters,
    simulate_device,
)

BASE = dict(
    switching_frequency_khz=8.0,
    dv_dt_v_per_us=3500.0,
    cable_length_m=25.0,
    shielding_quality=0.6,
    load_current_a=45.0,
    pwm_modulation_type="SPWM",
)
FIXED_SEED = 12345


def analyse(seed=FIXED_SEED, **overrides):
    params = DeviceParameters(**{**BASE, **overrides})
    bundle = extract_features(simulate_device(params, seed=seed))
    return (
        np.array([b.peak_dbuv for b in bundle.bands]),
        band_margins(bundle),
    )


def sweep(key, values, tolerance_db=1.0):
    print(f"\n--- {key} (expect non-decreasing peaks) ---")
    print(f"{'value':>12s} | {'peak A':>7s} {'peak B':>7s} {'peak C':>7s} | "
          f"{'marg A':>7s} {'marg B':>7s} {'marg C':>7s}")
    rows = []
    for v in values:
        peaks, margins = analyse(**{key: v})
        rows.append(peaks)
        vs = f"{v}" if isinstance(v, str) else f"{v:.4g}"
        print(f"{vs:>12s} | " + " ".join(f"{p:7.1f}" for p in peaks)
              + " | " + " ".join(f"{m:7.1f}" for m in margins))
    arr = np.array(rows)
    for i, band in enumerate(EMC_BANDS):
        steps = np.diff(arr[:, i])
        ok = bool(np.all(steps >= -tolerance_db))
        print(f"    {band.label:18s} {'OK ' if ok else 'FAIL'} "
              f"total {arr[-1, i] - arr[0, i]:+6.1f} dB, "
              f"worst step {steps.min():+.2f} dB")


def pwm_ordering(n_designs=150):
    """Verify PWM_CM_PENALTY_DB against measured band peaks.

    Each strategy is evaluated on the *same* set of random designs so the
    comparison is paired. Strategies sharing a penalty value are pooled, and the
    pooled means must be non-decreasing in the penalty.
    """
    from simulate import PWM_CM_PENALTY_DB

    print(f"\n--- PWM strategy ranking ({n_designs} paired random designs each) ---")
    table = {}
    for name in PWM_MODULATION_TYPES:
        rng = np.random.default_rng(2024)  # same designs for every strategy
        peaks = []
        for _ in range(n_designs):
            p = sample_random_parameters(rng)
            p = DeviceParameters(**{**p.as_dict(), "pwm_modulation_type": name})
            bundle = extract_features(simulate_device(p))
            peaks.append([b.peak_dbuv for b in bundle.bands])
        table[name] = np.mean(peaks, axis=0)

    ordered = sorted(PWM_MODULATION_TYPES, key=lambda n: PWM_CM_PENALTY_DB[n])
    print(f"{'strategy':>14s} {'penalty':>8s} | "
          f"{'peak A':>7s} {'peak B':>7s} {'peak C':>7s} | vs SPWM (dB)")
    for name in ordered:
        delta = table[name] - table["SPWM"]
        print(f"{name:>14s} {PWM_CM_PENALTY_DB[name]:>+8.1f} | "
              + " ".join(f"{v:7.2f}" for v in table[name])
              + " | " + " ".join(f"{d:+6.2f}" for d in delta))

    # Pool strategies that share a penalty value, then check the trend.
    penalties = sorted(set(PWM_CM_PENALTY_DB.values()))
    pooled = np.array([
        np.mean([table[n] for n in PWM_MODULATION_TYPES
                 if PWM_CM_PENALTY_DB[n] == p], axis=0)
        for p in penalties
    ])
    for i, band in enumerate(EMC_BANDS):
        steps = np.diff(pooled[:, i])
        ok = bool(np.all(steps >= -0.05))
        print(f"    {band.label:18s} {'OK ' if ok else 'FAIL'} "
              f"pooled means non-decreasing in penalty, worst step {steps.min():+.2f} dB")


def population(n=300):
    print(f"\n--- random population ({n} designs, per-design seeds) ---")
    rng = np.random.default_rng(7)
    margins = []
    for _ in range(n):
        bundle = extract_features(simulate_device(sample_random_parameters(rng)))
        margins.append(band_margins(bundle))
    margins = np.array(margins)
    fails = margins <= 0
    for i, band in enumerate(EMC_BANDS):
        print(f"{band.label:18s} margin p5/p50/p95 = "
              f"{np.percentile(margins[:, i], 5):7.1f} / "
              f"{np.percentile(margins[:, i], 50):7.1f} / "
              f"{np.percentile(margins[:, i], 95):7.1f}   "
              f"fail rate = {fails[:, i].mean():.1%}")
    print(f"any-band fail rate = {fails.any(axis=1).mean():.1%}")


def main():
    t0 = time.time()
    sweep("switching_frequency_khz", [2, 4, 6, 8, 12, 16, 20])
    sweep("dv_dt_v_per_us", [500, 1000, 2000, 3500, 6000, 10000])
    sweep("cable_length_m", [1, 3, 10, 25, 60, 120])
    sweep("load_current_a", [5, 15, 45, 100, 200])
    sweep("shielding_quality", [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    pwm_ordering()
    population()
    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
