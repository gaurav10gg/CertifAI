"""Closed-form design trade-off proxies and a switching-frequency sweep.

This is not another trained model. Two engineering approximations are evaluated
alongside the existing EMC risk score while ``switching_frequency_khz`` is swept
and every other design parameter is held fixed.

The two costs are the ones named by the "drop the carrier" countermeasure:
audible noise and motor current ripple. Switching loss is *not* included —
along a carrier sweep it moves with EMC headroom, not against it, so charting
it as a Pareto frontier would misrepresent the model.

Motor current ripple (ripple cost)
----------------------------------
Worst-case peak-to-peak current ripple of a two-level voltage-source inverter
into an inductive load, at duty cycle 0.5 (Mohan, Undeland & Robbins, *Power
Electronics*, ch. 8; Erickson & Maksimovic, *Fundamentals of Power
Electronics*, inductor-current ripple under PWM):

    ΔI_pp = (V_dc × d × (1 − d)) / (L_motor × f_sw)
           = V_dc / (4 × L_motor × f_sw)     at d = 0.5

Ripple therefore rises as the carrier falls. That is the EMC trade-off: fewer
commutations in the 150 kHz–30 MHz band, more current ripple at the motor.

``V_dc`` is the tool's fixed DC link ``simulate.V_DC_LINK`` (565 V).
``L_motor`` is the same assumed stator inductance used by the power-quality
simulator, ``simulate.MOTOR_L_H`` (6 mH) — a typical PM elevator-machine
stator, held constant so this proxy is not confounded with a load-dependent
inductance. It is not a user-facing parameter.

    ripple_cost = 100 × tanh(ΔI_pp / I_REF)

``I_REF = 5 A`` is a documented scale, not a measured ripple: 8 kHz on this
L and V_dc gives ΔI_pp ≈ 2.9 A, which maps near mid-scale, and 3 kHz does
not saturate.

This is **not** a time-domain current simulation of the assessed design's
torque ripple or RMS heating — it is the closed-form PWM inductor-ripple
envelope.

Audible-noise proxy (acoustic risk)
-----------------------------------
PWM carrier and its sidebands become audible as they enter ~20 Hz–20 kHz.
Industrial-drive practice treats carriers below ~8 kHz as a clear whine and
~16 kHz as "silent" / ultrasonic for most adult listeners (see e.g. ABB /
Siemens application notes on ultrasonic PWM; ISO 226 equal-loudness contours
for the steep rise of hearing sensitivity toward 1–4 kHz). This tool's
admissible carrier range is already 3–16 kHz, so the whole sweep sits in or
at the edge of the audible band.

    acoustic_risk = 100 / (1 + exp((f_kHz − 8.0) / 2.5))

A logistic centred at 8 kHz, scale 2.5 kHz: high at 3 kHz, ~50 at 8 kHz, low
at 16 kHz. This is **not** an acoustic simulation — no SPL, no cabin
coupling, no A-weighting.

Sweep
-----
The EMC number at each point is the same risk score the dashboard uses
(``predictor._compliance_score`` on the three band-margin regressors), but
SHAP, countermeasures and the spectrum payload are skipped so a 27-point
sweep stays interactive. The simulator noise seed is taken from the *held*
parameters (carrier replaced by a sentinel) so redrawn noise cannot fake a
non-monotonic EMC-vs-frequency curve. Clicking a point in the UI and running
a full ``/predict`` uses that design's own seed and may differ by the
documented seed scatter (~0.25 dB).
"""

from __future__ import annotations

import copy
import math
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from features import EMC_BANDS, extract_features
from predictor import (
    ModelBundle,
    ModelNotTrainedError,
    _compliance_score,
    load_models,
)
from simulate import (
    MOTOR_L_H,
    PARAMETER_RANGE_BY_KEY,
    V_DC_LINK,
    DeviceParameters,
    simulate_device,
)

# Carrier range of the tool. Sweeps default to the full interval.
F_SW_MIN_KHZ: float = PARAMETER_RANGE_BY_KEY["switching_frequency_khz"].minimum
F_SW_MAX_KHZ: float = PARAMETER_RANGE_BY_KEY["switching_frequency_khz"].maximum
DEFAULT_N_POINTS: int = 27

# Sentinel carrier used only to freeze the noise seed while f_sw is swept.
_SEED_SENTINEL_KHZ: float = 8.0

# Same stator inductance the PQ simulator uses. Documented in the module docstring.
L_MOTOR_H: float = MOTOR_L_H

# Scale for the tanh map of ΔI_pp. Documented in the module docstring.
RIPPLE_REF_A: float = 5.0

# Logistic acoustic proxy. Documented in the module docstring.
F_AUDIBLE_CENTER_KHZ: float = 8.0
F_AUDIBLE_SCALE_KHZ: float = 2.5

# "Drop the carrier to around 4 kHz" — the countermeasure text in predictor.py.
COUNTERMEASURE_TARGET_KHZ: float = 4.0

CAPTION: str = (
    "Trade-offs shown reflect the two costs named in this report's own "
    "recommendations: audible noise and motor current ripple, both of which "
    "rise as switching frequency is lowered to improve EMC margin. Switching "
    "frequency alone swept; all other parameters held at current values. "
    "Acoustic and ripple metrics are simplified engineering proxies, not "
    "simulated or measured quantities."
)

RIPPLE_FORMULA: str = (
    "ΔI_pp = V_dc / (4 × L_motor × f_sw) at duty 0.5 "
    f"(V_dc = {V_DC_LINK:.0f} V, L_motor = {L_MOTOR_H * 1e3:.1f} mH). "
    f"ripple_cost = 100 × tanh(ΔI_pp / {RIPPLE_REF_A:.0f} A). "
    "Two-level VSI inductor ripple (Mohan / Erickson); not a torque simulation."
)

ACOUSTIC_FORMULA: str = (
    f"acoustic_risk = 100 / (1 + exp((f_kHz − {F_AUDIBLE_CENTER_KHZ:.0f}) / "
    f"{F_AUDIBLE_SCALE_KHZ})). Logistic audible-band proxy; not an SPL simulation."
)


def ripple_peak_to_peak_a(switching_frequency_khz: float) -> float:
    """Worst-case PWM current ripple in amperes. See module docstring."""
    f_hz = float(switching_frequency_khz) * 1e3
    return V_DC_LINK / (4.0 * L_MOTOR_H * f_hz)


def ripple_cost(switching_frequency_khz: float) -> float:
    """0-100 map of :func:`ripple_peak_to_peak_a`. Higher = more current ripple."""
    delta_i = ripple_peak_to_peak_a(switching_frequency_khz)
    return float(100.0 * math.tanh(delta_i / RIPPLE_REF_A))


def acoustic_risk(switching_frequency_khz: float) -> float:
    """0-100 audible-band proxy. Higher = more likely to be heard. See docstring."""
    z = (float(switching_frequency_khz) - F_AUDIBLE_CENTER_KHZ) / F_AUDIBLE_SCALE_KHZ
    return float(100.0 / (1.0 + math.exp(z)))


def pareto_mask(
    scores: Sequence[float],
    costs: Sequence[float],
) -> List[bool]:
    """True where no other point is better on score *and* cost.

    Score is maximised (EMC headroom). Cost is minimised (ripple or acoustic).
    A point ``i`` is dominated if some ``j`` has score_j ≥ score_i and
    cost_j ≤ cost_i, with at least one inequality strict.
    """
    score_arr = np.asarray(scores, dtype=float)
    cost_arr = np.asarray(costs, dtype=float)
    n = score_arr.size
    flags = [True] * n
    for i in range(n):
        better_or_equal = (score_arr >= score_arr[i]) & (cost_arr <= cost_arr[i])
        strictly = (score_arr > score_arr[i]) | (cost_arr < cost_arr[i])
        better_or_equal[i] = False
        if np.any(better_or_equal & strictly):
            flags[i] = False
    return flags


def _with_frequency(params: DeviceParameters, f_khz: float) -> DeviceParameters:
    payload = params.as_dict()
    payload["switching_frequency_khz"] = float(f_khz)
    return DeviceParameters.clamped(**payload)


def _held_seed(params: DeviceParameters) -> int:
    """Noise seed that does not change as the carrier is swept."""
    payload = params.as_dict()
    payload["switching_frequency_khz"] = _SEED_SENTINEL_KHZ
    return DeviceParameters.clamped(**payload).deterministic_seed()


def _emc_at(
    params: DeviceParameters,
    models: ModelBundle,
    seed: int,
) -> Tuple[float, List[float]]:
    simulation = simulate_device(params, seed=seed)
    extracted = extract_features(simulation)
    scaled = models.scaler.transform(extracted.feature_vector.reshape(1, -1))
    margins = [float(reg.predict(scaled)[0]) for reg in models.regressors]
    return _compliance_score(margins), margins


def _trend(values: Sequence[float], *, atol: float = 1e-4) -> str:
    diffs = np.diff(np.asarray(values, dtype=float))
    n_up = int(np.sum(diffs > atol))
    n_down = int(np.sum(diffs < -atol))
    if n_down == 0 and n_up > 0:
        return "increasing"
    if n_up == 0 and n_down > 0:
        return "decreasing"
    if n_up == 0 and n_down == 0:
        return "flat"
    return "non_monotonic"


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation; no SciPy dependency."""
    rx = np.argsort(np.argsort(np.asarray(xs, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(ys, dtype=float)))
    if rx.size < 2:
        return 0.0
    corr = np.corrcoef(rx, ry)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _inversions(
    frequencies: Sequence[float],
    values: Sequence[float],
    *,
    expect: str,
    atol: float = 0.05,
) -> List[Dict[str, float]]:
    """Adjacent steps that move against ``expect`` ('increasing' or 'decreasing')."""
    out: List[Dict[str, float]] = []
    for f0, f1, v0, v1 in zip(frequencies[:-1], frequencies[1:], values[:-1], values[1:]):
        delta = float(v1) - float(v0)
        illegal = (
            (expect == "decreasing" and delta > atol)
            or (expect == "increasing" and delta < -atol)
        )
        if illegal:
            out.append(
                {
                    "from_khz": float(f0),
                    "to_khz": float(f1),
                    "from_value": float(v0),
                    "to_value": float(v1),
                    "delta": round(delta, 3),
                }
            )
    return out


def _nearest_index(grid: Sequence[float], target: float) -> int:
    arr = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(arr - target)))


def _sweep_cache_key(
    params: DeviceParameters,
    f_lo: float,
    f_hi: float,
    n_points: int,
) -> Tuple[Any, ...]:
    held = tuple(
        sorted(
            (key, round(float(value), 6) if isinstance(value, (int, float)) else value)
            for key, value in params.as_dict().items()
            if key != "switching_frequency_khz"
        )
    )
    return (held, round(f_lo, 3), round(f_hi, 3), int(n_points))


def _stamp_current(result: Dict[str, Any], current_khz: float) -> Dict[str, Any]:
    points = result.get("points") or []
    if not points:
        return result
    grid = [float(p["switching_frequency_khz"]) for p in points]
    current_index = _nearest_index(grid, current_khz)
    for index, point in enumerate(points):
        point["is_current"] = index == current_index
    return result


_SWEEP_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_SWEEP_CACHE_LOCK = Lock()


def sweep_switching_frequency(
    params: DeviceParameters,
    *,
    f_min_khz: float = F_SW_MIN_KHZ,
    f_max_khz: float = F_SW_MAX_KHZ,
    n_points: int = DEFAULT_N_POINTS,
    bundle: Optional[ModelBundle] = None,
) -> Dict[str, Any]:
    """Sweep carrier frequency; return points, Pareto flags and diagnostics."""
    if bundle is None:
        bundle = load_models()

    f_lo = float(np.clip(min(f_min_khz, f_max_khz), F_SW_MIN_KHZ, F_SW_MAX_KHZ))
    f_hi = float(np.clip(max(f_min_khz, f_max_khz), F_SW_MIN_KHZ, F_SW_MAX_KHZ))
    if f_hi <= f_lo:
        raise ValueError("switching-frequency sweep range must be non-empty")
    n_points = int(n_points)
    if n_points < 3:
        raise ValueError("n_points must be at least 3")

    cache_key = _sweep_cache_key(params, f_lo, f_hi, n_points)
    with _SWEEP_CACHE_LOCK:
        cached = _SWEEP_CACHE.get(cache_key)
    if cached is not None:
        return _stamp_current(copy.deepcopy(cached), params.switching_frequency_khz)

    grid = [float(v) for v in np.linspace(f_lo, f_hi, n_points)]
    seed = _held_seed(params)

    points: List[Dict[str, Any]] = []
    for f_khz in grid:
        candidate = _with_frequency(params, f_khz)
        score, margins = _emc_at(candidate, bundle, seed)
        delta_i = ripple_peak_to_peak_a(f_khz)
        points.append(
            {
                "switching_frequency_khz": round(f_khz, 3),
                "emc_risk_score": round(score, 2),
                "ripple_cost": round(ripple_cost(f_khz), 2),
                "acoustic_risk": round(acoustic_risk(f_khz), 2),
                "ripple_peak_to_peak_a": round(delta_i, 3),
                "predicted_margins_db": {
                    band.key: round(margin, 3)
                    for band, margin in zip(EMC_BANDS, margins)
                },
            }
        )

    emc_scores = [p["emc_risk_score"] for p in points]
    ripple_costs = [p["ripple_cost"] for p in points]
    ac_costs = [p["acoustic_risk"] for p in points]
    pareto_ripple = pareto_mask(emc_scores, ripple_costs)
    pareto_ac = pareto_mask(emc_scores, ac_costs)

    for index, point in enumerate(points):
        point["pareto_ripple"] = pareto_ripple[index]
        point["pareto_acoustic"] = pareto_ac[index]
        point["is_current"] = False

    emc_trend = _trend(emc_scores)
    ripple_trend = _trend(ripple_costs)
    ac_trend = _trend(ac_costs)
    rho_emc = _spearman(grid, emc_scores)
    rho_ripple = _spearman(grid, ripple_costs)
    rho_ac = _spearman(grid, ac_costs)
    emc_inversions = _inversions(grid, emc_scores, expect="decreasing")

    # Countermeasure #3: dropping the carrier toward 4 kHz should raise every
    # band's predicted margin relative to the top of the sweep.
    idx_low = _nearest_index(grid, COUNTERMEASURE_TARGET_KHZ)
    idx_high = len(grid) - 1
    low_margins = points[idx_low]["predicted_margins_db"]
    high_margins = points[idx_high]["predicted_margins_db"]
    band_deltas = {
        key: round(low_margins[key] - high_margins[key], 3) for key in low_margins
    }
    every_band_improves = all(delta > 0.0 for delta in band_deltas.values())
    emc_improves_at_low_carrier = (
        points[idx_low]["emc_risk_score"] > points[idx_high]["emc_risk_score"]
    )

    # Score is maximised, costs are minimised. Opposite-sign Spearman vs
    # frequency means both *goodness* directions move together (no trade-off).
    # Same-sign Spearman means one goodness rises while the other falls.
    n_pareto_ripple = sum(pareto_ripple)
    n_pareto_ac = sum(pareto_ac)
    ripple_trades_off = rho_emc * rho_ripple > 0.0
    acoustic_trades_off = rho_emc * rho_ac > 0.0

    payload = {
        "caption": CAPTION,
        "formulas": {
            "ripple_cost": RIPPLE_FORMULA,
            "acoustic_risk": ACOUSTIC_FORMULA,
            "ripple_ref_a": RIPPLE_REF_A,
            "l_motor_h": L_MOTOR_H,
            "v_dc_v": V_DC_LINK,
            "audible_center_khz": F_AUDIBLE_CENTER_KHZ,
            "audible_scale_khz": F_AUDIBLE_SCALE_KHZ,
        },
        "held": {
            key: value
            for key, value in params.as_dict().items()
            if key != "switching_frequency_khz"
        },
        "sweep": {
            "f_min_khz": round(f_lo, 3),
            "f_max_khz": round(f_hi, 3),
            "n_points": n_points,
            "seed": seed,
        },
        "points": points,
        "diagnostics": {
            "emc_risk_score_vs_frequency": emc_trend,
            "ripple_cost_vs_frequency": ripple_trend,
            "acoustic_risk_vs_frequency": ac_trend,
            "spearman_emc_vs_frequency": round(rho_emc, 3),
            "spearman_ripple_vs_frequency": round(rho_ripple, 3),
            "spearman_acoustic_vs_frequency": round(rho_ac, 3),
            "expected_emc_trend": "decreasing",
            "expected_ripple_trend": "decreasing",
            "expected_acoustic_trend": "decreasing",
            "emc_matches_expectation": rho_emc < 0.0,
            "ripple_matches_expectation": ripple_trend == "decreasing",
            "acoustic_matches_expectation": ac_trend == "decreasing",
            "emc_local_inversions": emc_inversions,
            "countermeasure_target_khz": points[idx_low]["switching_frequency_khz"],
            "countermeasure_reference_khz": points[idx_high]["switching_frequency_khz"],
            "margin_gain_db_at_4khz_vs_fmax": band_deltas,
            "every_band_improves_at_lower_carrier": every_band_improves,
            "emc_score_improves_at_lower_carrier": emc_improves_at_low_carrier,
            "agrees_with_carrier_countermeasure": (
                every_band_improves and emc_improves_at_low_carrier
            ),
            "n_pareto_ripple": n_pareto_ripple,
            "n_pareto_acoustic": n_pareto_ac,
            "ripple_trades_off_with_emc": ripple_trades_off,
            "acoustic_trades_off_with_emc": acoustic_trades_off,
            "pareto_note": (
                "Ripple cost and acoustic risk both rise as carrier falls, "
                "so each trades off against EMC headroom. Local EMC wiggles "
                "from PWM harmonic placement can dominate a few mid-sweep "
                "points; the 4 kHz vs 16 kHz endpoint comparison is the "
                "countermeasure check."
                if ripple_trades_off and acoustic_trades_off
                else "One or both cost proxies do not trade off against EMC "
                "along this sweep; do not chart a degenerate axis."
            ),
        },
    }
    with _SWEEP_CACHE_LOCK:
        _SWEEP_CACHE[cache_key] = copy.deepcopy(payload)
    return _stamp_current(payload, params.switching_frequency_khz)


def run_tradeoff(
    params: DeviceParameters,
    *,
    f_min_khz: float = F_SW_MIN_KHZ,
    f_max_khz: float = F_SW_MAX_KHZ,
    n_points: int = DEFAULT_N_POINTS,
) -> Dict[str, Any]:
    """Public entry used by the API. Raises ``ModelNotTrainedError`` if needed."""
    if not isinstance(params, DeviceParameters):
        raise TypeError("params must be a DeviceParameters instance")
    try:
        return sweep_switching_frequency(
            params,
            f_min_khz=f_min_khz,
            f_max_khz=f_max_khz,
            n_points=n_points,
        )
    except ModelNotTrainedError:
        raise
