"""
predictor.py -- Inference engine: physics chain + models -> assessment result.

This is the single place where a set of device parameters becomes the assessment
object the API returns and the PDF renders. Keeping it out of ``app.py`` means the
scoring logic can be exercised directly from a script or a test without an HTTP
server in the way.

Flow
----
    parameters
      -> simulate.py        (time-domain common-mode disturbance)
      -> features.py        (FFT, receiver emulation, band features, margins)
      -> XGBoost models     (P(fail) and margin-to-limit per band)
      -> scoring            (compliance score, confidence, risk attribution)

Both the model's prediction and the simulator's own measurement are returned for
every band. They should agree closely; surfacing both is deliberate, because a
tool that hides its own disagreement is not auditable.

Note on ordering: this endpoint runs the simulation *and* the models, so the
models are not saving any work here -- they contribute calibrated probabilities, a
margin estimate with a measured error bar, and exact SHAP attribution for the risk
factor. The models' standalone value is in the design-only configuration measured
by ``train_model.design_only_ablation``, which predicts compliance from the six
design parameters at ~1-2 dB margin error without simulating at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import xgboost as xgb

from features import (
    DESIGN_FEATURE_SPECS,
    EMC_BANDS,
    FEATURE_NAMES,
    LIMIT_CURVE_ANCHORS,
    LIMIT_CURVE_DESCRIPTION,
    MONOTONE_CONSTRAINTS_RISK,
    FeatureBundle,
    extract_features,
    parameter_risk_position,
)
from simulate import (
    PARAMETER_RANGE_BY_KEY,
    PWM_CM_PENALTY_DB,
    PWM_MODULATION_LABELS,
    SWITCHING_DEVICE_LABELS,
    DeviceParameters,
    simulate_device,
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "certifai_models.joblib"
RADIATED_MODEL_PATH = Path(__file__).resolve().parent / "models" / "radiated_models.joblib"

# Compliance score mapping. A band exactly on the limit (0 dB margin) scores 50;
# the tanh gives a smooth, bounded curve so a design 30 dB clear does not read as
# meaningfully different from one 40 dB clear.
SCORE_MARGIN_SCALE_DB: float = 8.0

# The overall score blends the mean band score with the worst band score, so a
# single badly failing band cannot be averaged away by two comfortable ones.
WORST_BAND_WEIGHT: float = 0.5

CONFIDENCE_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (80.0, "High Confidence"),
    (60.0, "Moderate Confidence"),
    (0.0, "Low Confidence"),
)

# Confidence is capped below 100 on purpose. The interval around the risk score
# is the quantity that should move; this figure is retained as a secondary
# "how close is this to a band edge" indicator, not a claim about hardware.
CONFIDENCE_CEILING: float = 97.0

CONFIDENCE_CEILING_NOTE: str = (
    "The ± band on the risk score is the spread across an ensemble of "
    "monotone-constrained models (or, if the ensemble has not been trained, a "
    "conservative mapping of held-out margin error). It measures agreement with "
    "this tool's own physics simulation, not with accredited lab measurements, "
    "and is therefore a floor on the true uncertainty."
)

DISAGREEMENT_PENALTY: float = 0.80

RISK_FRAMING: str = (
    "This tool estimates EMC risk from simulated physics. It does not predict "
    "EN 12016 certification outcomes, which require accredited lab measurement."
)

DISCLAIMER_SHORT: str = (
    "This is a simulation-based pre-compliance risk indicator to guide design "
    "decisions, not a substitute for accredited EMC lab certification."
)

DISCLAIMER_LONG: str = (
    "This is a simulation-based pre-compliance risk assessment intended to "
    "support early-stage design decisions. It does not replace accredited EMC "
    "certification testing (e.g. per EN 12016), and it does not predict "
    "certification outcomes."
)

SCOPE_STATEMENT: str = (
    "Conducted emissions (150 kHz-30 MHz) use a validated common-mode circuit "
    "model, with the same simulation-consistency validation as the rest of this "
    "tool. Radiated emissions (30 MHz-1 GHz) use a separate, more exploratory "
    "model based on clock-harmonic and loop-radiation theory — treat this score "
    "as a rough directional indicator only. Immunity and functional-safety "
    "remain fully out of scope."
)

RADIATED_CAPTION: str = (
    "Lower confidence than the conducted score above — this model has less "
    "validation history in this tool."
)

RADIATED_DISCLAIMER: str = (
    "This radiated-emissions estimate uses each band's own peak level as one "
    "of its inputs, so its low error describes agreement with this tool's own "
    "simulation, not a chamber measurement. The limit line shown is a synthetic "
    "step anchored on publicly described CISPR 11 Group 1 Class A values at "
    "10 m — it is not the normative EN 12016 limit."
)

RISK_TIERS: Tuple[Tuple[float, str, str], ...] = (
    (70.0, "LOW", "Low risk"),
    (40.0, "MODERATE", "Moderate risk"),
    (0.0, "HIGH", "High risk"),
)


class ModelNotTrainedError(RuntimeError):
    """Raised when the saved model artifacts are missing."""


@dataclass(frozen=True)
class ModelBundle:
    """Loaded artifacts plus the metadata needed to interpret them."""

    scaler: Any
    classifiers: Sequence[xgb.XGBClassifier]
    regressors: Sequence[xgb.XGBRegressor]
    # Optional bootstrap ensemble of margin regressors, shape (n_members, n_bands).
    ensemble_regressors: Sequence[Sequence[xgb.XGBRegressor]]
    # Six-parameter margin regressors (no spectral features). Used for SHAP.
    design_only_scaler: Any
    design_only_regressors: Sequence[xgb.XGBRegressor]
    design_only_feature_names: Sequence[str]
    feature_names: Sequence[str]
    # Held-out margin RMSE of the deployed models against the simulator.
    margin_rmse_db: Sequence[float]
    # The wider figure actually used for confidence -- see _conservative_rmse.
    margin_rmse_conservative_db: Sequence[float]
    report: Dict[str, Any]
    artifact_version: str


def _conservative_rmse(report: Dict[str, Any], model_rmse: Sequence[float]) -> List[float]:
    """Per-band margin error to use when quoting confidence.

    The deployed models reproduce the simulator to within ~0.3 dB, so using that
    figure directly makes every confidence score pin to the ceiling and tells the
    user nothing. Worse, it would be quietly overstating what is known: 0.3 dB is
    how well the models agree with *our own simulation*, not how well either agrees
    with hardware.

    Instead we take the larger of two measured quantities per band:

      * the deployed models' held-out margin RMSE, and
      * the design-only ablation's margin RMSE (1.4-2.5 dB), which is what the
        error becomes when the spectral features are removed and the spectrum has
        to be inferred from the design alone.

    The second is a measurable proxy for how much the answer depends on modelling
    choices rather than on the data, so it is the more honest error bar to quote.
    It is still a floor on the true uncertainty, not an estimate of it, because the
    simulator-versus-reality term remains unmeasured.
    """
    ablation = (report.get("design_only_ablation") or {}).get("band_metrics", [])
    ablation_rmse = [row.get("margin_rmse_db") for row in ablation]

    conservative: List[float] = []
    for index, value in enumerate(model_rmse):
        candidates = [float(value)]
        if index < len(ablation_rmse) and ablation_rmse[index] is not None:
            candidates.append(float(ablation_rmse[index]))
        conservative.append(max(candidates))
    return conservative


@lru_cache(maxsize=1)
def load_models(path: Optional[str] = None) -> ModelBundle:
    """Load and cache the trained artifacts."""
    model_path = Path(path) if path else MODEL_PATH
    if not model_path.exists():
        raise ModelNotTrainedError(
            f"No trained models at {model_path}. Run: python train_model.py"
        )

    raw = joblib.load(model_path)
    if list(raw["feature_names"]) != list(FEATURE_NAMES):
        raise ModelNotTrainedError(
            "Saved model feature schema does not match features.FEATURE_NAMES. "
            "The feature definition changed since training -- retrain with "
            "python train_model.py"
        )

    report = raw.get("report", {})
    return ModelBundle(
        scaler=raw["scaler"],
        classifiers=raw["classifiers"],
        regressors=raw["regressors"],
        ensemble_regressors=raw.get("ensemble_regressors") or (),
        design_only_scaler=raw.get("design_only_scaler"),
        design_only_regressors=raw.get("design_only_regressors") or (),
        design_only_feature_names=raw.get("design_only_feature_names") or (),
        feature_names=raw["feature_names"],
        margin_rmse_db=raw["margin_rmse_db"],
        margin_rmse_conservative_db=_conservative_rmse(report, raw["margin_rmse_db"]),
        report=report,
        artifact_version=raw.get("artifact_version", "unknown"),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _band_score(margin_db: float) -> float:
    """Map a margin in dB onto 0-100, with 0 dB margin scoring exactly 50."""
    return float(
        np.clip(50.0 + 50.0 * math.tanh(margin_db / SCORE_MARGIN_SCALE_DB), 0.0, 100.0)
    )


def _compliance_score(margins_db: Sequence[float]) -> float:
    scores = [_band_score(m) for m in margins_db]
    blended = (
        (1.0 - WORST_BAND_WEIGHT) * float(np.mean(scores))
        + WORST_BAND_WEIGHT * float(np.min(scores))
    )
    return float(np.clip(blended, 0.0, 100.0))


def _risk_level(score: float) -> Dict[str, str]:
    """Three-tier summary of the 0-100 risk score.

    Higher score = more headroom = lower EMC risk. The tiers are a summary of
    the per-band margins, not a certification prediction.
    """
    for threshold, key, label in RISK_TIERS:
        if score >= threshold:
            return {"key": key, "label": label}
    return {"key": "HIGH", "label": "High risk"}


def _risk_copy(level_key: str) -> str:
    if level_key == "LOW":
        return (
            "Simulated emissions sit comfortably below the assumed limit in every "
            "band. Treat this as design headroom, not as a certification result."
        )
    if level_key == "MODERATE":
        return (
            "At least one band is close to the assumed limit, or the overall "
            "headroom is thin. A targeted change is worth evaluating before a lab."
        )
    return (
        "One or more bands are predicted to exceed the assumed limit by a "
        "material margin. This is a design-risk flag, not a lab fail."
    )


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _confidence(
    predicted_margins: Sequence[float],
    fail_probabilities: Sequence[float],
    margin_rmse_db: Sequence[float],
) -> float:
    """Probability that the pass/fail verdict is on the correct side of the limit.

    Treating the per-band margin error as Gaussian with the standard deviation from
    :func:`_conservative_rmse`, the chance a band's verdict is correct is
    Phi(|margin| / sigma): a design predicted 20 dB clear is essentially certain,
    whereas one predicted 0.5 dB clear is close to a coin toss.

    The overall verdict needs every band to be right, so the per-band
    probabilities are multiplied. Bands where the classifier and the regressor
    disagree about the sign are penalised further -- that disagreement is real
    information about model uncertainty and should not be smoothed over.
    """
    confidence = 1.0
    for margin, p_fail, rmse in zip(
        predicted_margins, fail_probabilities, margin_rmse_db
    ):
        sigma = max(float(rmse), 0.2)  # floor guards against over-claiming
        confidence *= _normal_cdf(abs(float(margin)) / sigma)
        regressor_says_fail = margin <= 0.0
        classifier_says_fail = p_fail >= 0.5
        if regressor_says_fail != classifier_says_fail:
            confidence *= DISAGREEMENT_PENALTY
    return float(np.clip(confidence * 100.0, 0.0, CONFIDENCE_CEILING))


def _confidence_label(confidence: float) -> str:
    for threshold, label in CONFIDENCE_THRESHOLDS:
        if confidence >= threshold:
            return label
    return CONFIDENCE_THRESHOLDS[-1][1]


# ---------------------------------------------------------------------------
# Risk attribution
# ---------------------------------------------------------------------------
_SUGGESTIONS: Dict[str, str] = {
    "switching_frequency_khz": (
        "Drop the carrier to around {target:.0f} kHz. Spectral line amplitude "
        "scales with commutation rate, so this lowers every band at once -- at the "
        "cost of more motor current ripple and audible noise."
    ),
    "dv_dt_v_per_us": (
        "Slow the switching edges to around {target:.0f} V/us with a larger gate "
        "resistor or an output dv/dt filter. This pulls the spectral envelope "
        "corner down and mainly helps the 5-30 MHz band."
    ),
    "cable_length_m": (
        "Shorten the motor cable run towards {target:.0f} m, or add an output "
        "sine/dv-dt filter at the drive. Cable capacitance to earth scales "
        "directly with length and is usually the largest single contributor."
    ),
    "shielding_quality": (
        "Improve the screen: 360-degree gland terminations at both ends, "
        "targeting an effectiveness of about {target:.0%}. This is normally "
        "the cheapest large improvement."
    ),
    "cm_choke_effectiveness": (
        "Fit a common-mode choke on the motor leads. A ferrite or toroid "
        "suppresses the same common-mode current this conducted score is built "
        "on. Aim for about {target:.0%} of a full 2 mH choke into the 50 ohm LISN."
    ),
    "load_current_a": (
        "Re-assess at a lower duty point, or de-rate towards {target:.0f} A. "
        "Emission amplitude follows return-path current, though only sub-linearly."
    ),
    "pwm_cm_penalty_db": (
        "Change modulation strategy. Discontinuous PWM (DPWM) clamps one leg at a "
        "time and removes roughly a third of all commutations, which lowers the "
        "measured level by about 2.5 dB in this model."
    ),
}

# How far to move each parameter when proposing a concrete target.
#
# Deliberately multiplicative for the parameters whose admissible range spans more
# than a decade (dv/dt, cable length, load current). A fixed fraction of the range
# would be meaningless for those: a quarter of the 1-120 m cable range is 30 m, so
# a 30 m cable would be told to shorten to 1 m. A proportional step keeps the
# suggestion realistic at every point in the range.
_MULTIPLICATIVE_STEP: Dict[str, float] = {
    "switching_frequency_khz": 0.75,
    "dv_dt_v_per_us": 0.60,
    "cable_length_m": 0.70,
    "load_current_a": 0.85,
}

# Shielding quality is a bounded 0-1 index, so an additive step is the natural
# expression of "improve it by a meaningful amount".
_SHIELDING_STEP: float = 0.15


def _suggested_target(key: str, current: float) -> float:
    """A concrete, in-range value to aim for, moving away from the risky end."""
    if key == "pwm_cm_penalty_db":
        return min(PWM_CM_PENALTY_DB.values())

    rge = PARAMETER_RANGE_BY_KEY[key]
    if key in ("shielding_quality", "cm_choke_effectiveness"):
        target = current + _SHIELDING_STEP
    else:
        target = current * _MULTIPLICATIVE_STEP[key]
    return float(np.clip(target, rge.minimum, rge.maximum))


def _shap_contributions(
    classifier: xgb.XGBClassifier, scaled_features: np.ndarray
) -> np.ndarray:
    """Exact per-feature SHAP contributions to the log-odds of failure.

    Used only for the ranked-countermeasures heuristic. The SHAP chart uses
    :func:`_shap_margin_contributions` on the design-only margin regressor.
    """
    try:
        matrix = xgb.DMatrix(scaled_features)
        contributions = classifier.get_booster().predict(
            matrix, pred_contribs=True, validate_features=False
        )
        return np.asarray(contributions)[0, : len(FEATURE_NAMES)]
    except Exception:  # pragma: no cover - defensive fallback
        importances = np.asarray(classifier.feature_importances_, dtype=float)
        return importances


def _design_only_matrix(bundle: ModelBundle, features: np.ndarray) -> np.ndarray:
    """Slice the 18-feature vector to the six design parameters and scale them."""
    names = list(bundle.design_only_feature_names) or [
        spec.name for spec in DESIGN_FEATURE_SPECS
    ]
    indices = [FEATURE_NAMES.index(name) for name in names]
    vector = np.asarray(features, dtype=float).reshape(-1)[indices].reshape(1, -1)
    if bundle.design_only_scaler is None:
        raise ModelNotTrainedError(
            "Design-only margin models are missing. Run: python train_model.py --design-only"
        )
    return bundle.design_only_scaler.transform(vector)


def _shap_margin_contributions(
    bundle: ModelBundle,
    features: np.ndarray,
    band_index: int,
) -> Tuple[np.ndarray, float, float]:
    """Exact SHAP of the design-only margin regressor for one band.

    Returns ``(shap_values, bias, predicted_margin_db)``. The additive identity
    ``sum(shap_values) + bias ≈ predicted_margin_db`` holds for tree SHAP.
    """
    if not bundle.design_only_regressors:
        raise ModelNotTrainedError(
            "Design-only margin models are missing. Run: python train_model.py --design-only"
        )
    scaled = _design_only_matrix(bundle, features)
    regressor = bundle.design_only_regressors[band_index]
    matrix = xgb.DMatrix(scaled)
    booster = regressor.get_booster()
    best = getattr(regressor, "best_iteration", None)
    kwargs: Dict[str, Any] = {"validate_features": False}
    if best is not None:
        kwargs["iteration_range"] = (0, int(best) + 1)
    predicted = float(booster.predict(matrix, **kwargs)[0])
    contributions = np.asarray(
        booster.predict(matrix, pred_contribs=True, **kwargs)
    )[0]
    n_features = scaled.shape[1]
    shap_values = np.asarray(contributions[:n_features], dtype=float)
    bias = float(contributions[n_features])
    return shap_values, bias, predicted


def _risk_factor(
    bundle: ModelBundle,
    features: np.ndarray,
    scaled: np.ndarray,
    params: DeviceParameters,
    band_index: int,
    band_fails: bool,
) -> Dict[str, Any]:
    """Name the design parameter most responsible for the worst band's outcome."""
    contributions = _shap_contributions(bundle.classifiers[band_index], scaled)

    ranked: List[Tuple[float, str]] = []
    for spec in DESIGN_FEATURE_SPECS:
        index = FEATURE_NAMES.index(spec.name)
        contribution = float(contributions[index])
        # Blend the model's attribution with how far the parameter sits towards
        # its own risky extreme. A parameter already at its safest value should
        # not be reported as the thing to change, even if the trees lean on it.
        position = parameter_risk_position(spec.name, float(features[index]))
        ranked.append((contribution * (0.35 + 0.65 * position), spec.name))

    ranked.sort(reverse=True)
    score, key = ranked[0]

    if key == "pwm_cm_penalty_db":
        label = "PWM Modulation Strategy"
        unit = ""
        current_display: Any = PWM_MODULATION_LABELS[params.pwm_modulation_type]
        current_value = float(params.pwm_cm_penalty_db)
    else:
        rge = PARAMETER_RANGE_BY_KEY[key]
        label = rge.label
        unit = rge.unit
        current_value = float(getattr(params, key))
        current_display = current_value

    target = _suggested_target(key, current_value)
    band = EMC_BANDS[band_index]

    if band_fails:
        statement = (
            f"{label} is the dominant contributor to the predicted "
            f"{band.label} exceedance."
        )
    else:
        statement = (
            f"{label} is the largest single contributor to emissions in "
            f"{band.label}, the band with the least headroom."
        )

    return {
        "parameter": key,
        "label": label,
        "unit": unit,
        "current_value": current_display,
        "suggested_value": (
            "DPWM" if key == "pwm_cm_penalty_db" else round(target, 2)
        ),
        "attribution_score": round(score, 4),
        "driving_band": band.key,
        "driving_band_label": band.label,
        "statement": statement,
        "suggestion": _SUGGESTIONS[key].format(target=target),
        "ranking": [
            {"parameter": name, "attribution_score": round(value, 4)}
            for value, name in ranked
        ],
    }


def _shap_waterfall(
    bundle: ModelBundle,
    features: np.ndarray,
    band_index: int,
) -> Dict[str, Any]:
    """Design-only SHAP contributions to predicted margin (dB) for the worst band."""
    shap_values, bias, predicted = _shap_margin_contributions(
        bundle, features, band_index
    )
    names = list(bundle.design_only_feature_names) or [
        spec.name for spec in DESIGN_FEATURE_SPECS
    ]
    rows: List[Dict[str, Any]] = []
    for name, value in zip(names, shap_values):
        if name == "pwm_cm_penalty_db":
            label = "PWM modulation"
        else:
            label = PARAMETER_RANGE_BY_KEY[name].label
        shap_db = float(value)
        rows.append({
            "parameter": name,
            "label": label,
            "shap": round(shap_db, 3),
            "raises_risk": shap_db < 0.0,
        })
    rows.sort(key=lambda row: abs(float(row["shap"])), reverse=True)
    band_label = EMC_BANDS[band_index].label
    return {
        "band": band_label,
        "unit": "dB",
        "base_value_db": round(bias, 3),
        "predicted_margin_db": round(predicted, 3),
        "note": (
            "Exact tree SHAP values from the design-only margin model "
            f"({len(names)} design parameters, no spectral shortcut) for "
            f"{band_label}, the band with the least predicted headroom. "
            "Positive values increase predicted margin (more headroom); "
            "negative values reduce it."
        ),
        "contributions": rows,
    }


def _countermeasures(
    ranking: Sequence[Dict[str, Any]],
    params: DeviceParameters,
    n: int = 3,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for entry in ranking[:n]:
        key = str(entry["parameter"])
        if key == "pwm_cm_penalty_db":
            current = float(params.pwm_cm_penalty_db)
            target = _suggested_target(key, current)
            items.append({
                "parameter": key,
                "label": "PWM Modulation Strategy",
                "suggestion": _SUGGESTIONS[key].format(target=target),
                "current_value": PWM_MODULATION_LABELS[params.pwm_modulation_type],
                "suggested_value": "DPWM",
            })
            continue
        rge = PARAMETER_RANGE_BY_KEY[key]
        current = float(getattr(params, key))
        target = _suggested_target(key, current)
        items.append({
            "parameter": key,
            "label": rge.label,
            "suggestion": _SUGGESTIONS[key].format(target=target),
            "current_value": round(current, 2),
            "suggested_value": round(target, 2),
        })
    return items


def _model_sensitivity(
    bundle: ModelBundle,
    features: np.ndarray,
    base_score: float,
) -> List[Dict[str, Any]]:
    """Local finite-difference of the risk score in feature space.

    Each design parameter is nudged a small fraction of its training-scale
    spread and the monotone-constrained regressors are re-evaluated -- no extra
    physics simulation. This answers "which lever moves the needle from here"
    at interactive latency. ``sensitivity_analysis`` below re-runs the simulator
    when a physics-level gradient is required.
    """
    scaled_base = bundle.scaler.transform(features.reshape(1, -1))
    raw = features.copy()
    rows: List[Dict[str, Any]] = []
    for spec in DESIGN_FEATURE_SPECS:
        index = FEATURE_NAMES.index(spec.name)
        scale = float(bundle.scaler.scale_[index]) if hasattr(bundle.scaler, "scale_") else 1.0
        delta = 0.25 * max(abs(scale), 1e-6) * spec.risk_sign
        nudged = raw.copy()
        nudged[index] += delta
        scaled = bundle.scaler.transform(nudged.reshape(1, -1))
        margins = [
            float(regressor.predict(scaled)[0]) for regressor in bundle.regressors
        ]
        score = _compliance_score(margins)
        if spec.name == "pwm_cm_penalty_db":
            label = "PWM modulation"
            unit = "dB"
        else:
            rge = PARAMETER_RANGE_BY_KEY[spec.name]
            label = rge.label
            unit = rge.unit
        rows.append({
            "parameter": spec.name,
            "label": label,
            "unit": unit,
            "delta_score": round(score - base_score, 2),
            "direction": "raises_risk" if (score - base_score) < 0 else "lowers_risk",
        })
    rows.sort(key=lambda row: abs(float(row["delta_score"])), reverse=True)
    _ = scaled_base
    return rows


def _ensemble_interval(
    bundle: ModelBundle, scaled: np.ndarray, point_score: float
) -> Tuple[float, float, float]:
    """Return (low, high, plus_minus) for the risk score."""
    members = bundle.ensemble_regressors
    scores: List[float] = [point_score]
    if members:
        for member in members:
            margins = [float(reg.predict(scaled)[0]) for reg in member]
            scores.append(_compliance_score(margins))
    else:
        # Fallback: map conservative margin RMSE through the score function.
        spread = float(np.mean(bundle.margin_rmse_conservative_db))
        scores.extend([
            _compliance_score([-spread]),  # dummy to size the band
        ])
        half = min(12.0, max(3.0, spread * 2.0))
        return (
            float(np.clip(point_score - half, 0, 100)),
            float(np.clip(point_score + half, 0, 100)),
            round(half, 1),
        )
    arr = np.asarray(scores, dtype=float)
    std = float(arr.std())
    half = float(np.clip(1.0 * std if std > 0.15 else 3.0, 2.0, 15.0))
    return (
        float(np.clip(point_score - half, 0, 100)),
        float(np.clip(point_score + half, 0, 100)),
        round(half, 1),
    )


def _serialize_signals(simulation: Any) -> List[Dict[str, Any]]:
    traces = []
    for trace in getattr(simulation, "explorer", ()) or ():
        traces.append({
            "key": trace.key,
            "label": trace.label,
            "unit": trace.unit,
            "timescale": trace.timescale,
            "description": trace.description,
            "time_ms": [round(float(t) * 1e3, 4) for t in trace.time_s],
            "values": [round(float(v), 4) for v in trace.values],
        })
    return traces


def _serialize_power_quality(extracted: FeatureBundle) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in extracted.power_quality:
        rows.append({
            "key": report.key,
            "label": report.label,
            "thd_percent": round(report.thd_percent, 1),
            "fundamental_hz": report.fundamental_hz,
            "dominant_orders": list(report.dominant_orders),
            "dominant_statement": report.dominant_statement,
            "peaks": [
                {
                    "order": peak.order,
                    "frequency_hz": round(peak.frequency_hz, 1),
                    "amplitude": round(peak.amplitude, 4),
                    "percent_of_fundamental": round(peak.percent_of_fundamental, 1),
                }
                for peak in report.peaks
            ],
        })
    return rows


# Drivers the radiated headline may name. Band peak levels are model inputs, so
# they are left out of this ranking; otherwise the "cause" would always be the
# peak the model was just shown.
_RADIATED_DRIVER_GROUPS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("clock_frequency_mhz", "Clock frequency", ("clock_frequency_mhz",)),
    ("di_dt_a_per_us", "di/dt", ("di_dt_a_per_us",)),
    (
        "cable_resonance",
        "Cable resonance",
        ("cable_length_m", "rad_low_resonance", "rad_high_resonance"),
    ),
)


@lru_cache(maxsize=1)
def load_radiated_models() -> Dict[str, Any]:
    if not RADIATED_MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"No radiated models at {RADIATED_MODEL_PATH}. "
            "Run: python tools/train_radiated.py"
        )
    return joblib.load(RADIATED_MODEL_PATH)


def _radiated_assessment(params: DeviceParameters) -> Dict[str, Any]:
    """Separate 30 MHz-1 GHz score. Never blended into the conducted score."""
    from radiated import (
        RADIATED_LIMIT_DESCRIPTION,
        radiated_features,
        radiated_spectrum,
    )

    bundle = load_radiated_models()
    vector, physics = radiated_features(params)
    spectrum = radiated_spectrum(params)
    scaled = bundle["scaler"].transform(vector.reshape(1, -1))
    predicted = [float(reg.predict(scaled)[0]) for reg in bundle["regressors"]]
    score = _compliance_score(predicted)
    level = _risk_level(score)
    worst = int(np.argmin(predicted))

    regressor = bundle["regressors"][worst]
    contrib = np.asarray(
        regressor.get_booster().predict(
            xgb.DMatrix(scaled), pred_contribs=True, validate_features=False
        )
    )[0]
    names = list(bundle["feature_names"])
    shap_by_name = {name: float(contrib[index]) for index, name in enumerate(names)}

    ranked: List[Tuple[float, str, str, float]] = []
    for key, label, members in _RADIATED_DRIVER_GROUPS:
        total = float(sum(shap_by_name.get(member, 0.0) for member in members))
        ranked.append((abs(total), key, label, total))
    ranked.sort(key=lambda row: row[0], reverse=True)
    _magnitude, top_key, top_label, top_shap = ranked[0]
    band_label = str(physics[worst]["label"])

    bands: List[Dict[str, Any]] = []
    mae = list(bundle.get("test_mae_db") or [0.0, 0.0])
    for index, row in enumerate(physics):
        margin = predicted[index]
        bands.append({
            "key": row["key"],
            "label": row["label"],
            "f_low_hz": row["f_low_hz"],
            "f_high_hz": row["f_high_hz"],
            "passes": bool(margin > 0.0),
            "fail_probability": 0.0,
            "predicted_margin_db": round(margin, 2),
            "band_score": round(_band_score(margin), 1),
            "margin_uncertainty_db": round(float(mae[index]), 2),
            "simulated_margin_db": round(float(row["margin_db"]), 2),
            "simulated_peak_dbuv": round(float(row["peak_dbuvm"]), 1),
            "simulated_peak_frequency_hz": round(float(row["peak_frequency_hz"]), 0),
            "limit_at_peak_dbuv": round(float(row["limit_dbuvm"]), 1),
            "worst_frequency_hz": round(float(row["peak_frequency_hz"]), 0),
            "harmonic_count": int(row["n_lines"]),
            "thd_score_db": 0.0,
        })

    return {
        "title": "Radiated Emissions Risk (Exploratory)",
        "badge": "EXPLORATORY",
        "caption": RADIATED_CAPTION,
        "disclaimer": RADIATED_DISCLAIMER,
        "risk_score": round(score, 1),
        "risk_level": level["key"],
        "risk_label": level["label"],
        "top_factor": {
            "key": top_key,
            "label": top_label,
            "shap_db": round(float(top_shap), 3),
            "band": band_label,
            "statement": (
                f"{top_label} ranks highest among clock frequency, di/dt and "
                f"cable resonance for {band_label} "
                f"({float(top_shap):+.2f} dB on the predicted margin). "
                f"The band peak, which this model also takes as an input, "
                f"accounts for most of the margin."
            ),
        },
        "bands": bands,
        "spectrum": {
            "frequency_hz": [round(float(f), 1) for f in spectrum["frequency_hz"]],
            "emission_dbuv": [round(float(v), 2) for v in spectrum["emission_dbuvm"]],
            "limit_dbuv": [round(float(v), 2) for v in spectrum["limit_dbuvm"]],
        },
        "limit_description": RADIATED_LIMIT_DESCRIPTION,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def predict(
    params: DeviceParameters,
    *,
    device_id: Optional[str] = None,
    device_name: Optional[str] = None,
    bundle: Optional[ModelBundle] = None,
) -> Dict[str, Any]:
    """Run the full assessment for one device configuration."""
    models = bundle or load_models()

    simulation = simulate_device(params)
    extracted: FeatureBundle = extract_features(simulation)

    features = extracted.feature_vector.reshape(1, -1)
    scaled = models.scaler.transform(features)

    fail_probabilities: List[float] = []
    predicted_margins: List[float] = []
    for classifier, regressor in zip(models.classifiers, models.regressors):
        fail_probabilities.append(float(classifier.predict_proba(scaled)[0, 1]))
        predicted_margins.append(float(regressor.predict(scaled)[0]))

    bands: List[Dict[str, Any]] = []
    for index, (band, analysis) in enumerate(zip(EMC_BANDS, extracted.bands)):
        predicted_margin = predicted_margins[index]
        bands.append({
            "key": band.key,
            "label": band.label,
            "f_low_hz": band.f_low_hz,
            "f_high_hz": band.f_high_hz,
            "passes": bool(predicted_margin > 0.0),
            "fail_probability": round(fail_probabilities[index], 4),
            "predicted_margin_db": round(predicted_margin, 2),
            "band_score": round(_band_score(predicted_margin), 1),
            "margin_uncertainty_db": round(
                float(models.margin_rmse_conservative_db[index]), 2
            ),
            # The simulator's own measurement, shown alongside the prediction so
            # the two can be compared rather than taken on trust.
            "simulated_margin_db": round(analysis.margin_db, 2),
            "simulated_peak_dbuv": round(analysis.peak_dbuv, 1),
            "simulated_peak_frequency_hz": round(analysis.peak_frequency_hz, 0),
            "limit_at_peak_dbuv": round(analysis.limit_at_peak_dbuv, 1),
            "worst_frequency_hz": round(analysis.worst_frequency_hz, 0),
            "harmonic_count": analysis.harmonic_count,
            "thd_score_db": round(analysis.thd_score, 2),
        })

    compliance_score = _compliance_score(predicted_margins)
    confidence = _confidence(
        predicted_margins, fail_probabilities, models.margin_rmse_conservative_db
    )
    level = _risk_level(compliance_score)
    score_low, score_high, plus_minus = _ensemble_interval(models, scaled, compliance_score)

    # The band with the least predicted headroom drives the narrative, whether or
    # not it actually sits above the assumed limit.
    worst_index = int(np.argmin(predicted_margins))
    risk_factor = _risk_factor(
        models, extracted.feature_vector, scaled, params,
        worst_index, not bands[worst_index]["passes"],
    )
    shap = _shap_waterfall(models, extracted.feature_vector, worst_index)
    measures = _countermeasures(risk_factor["ranking"], params, n=3)
    radiated = _radiated_assessment(params)
    sensitivity = _model_sensitivity(models, extracted.feature_vector, compliance_score)

    spectrum = extracted.spectrum
    consistency = _consistency_summary(models)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device_id": device_id,
        "device_name": device_name or "Custom Configuration",
        "parameters": params.as_dict(),
        "parameter_display": _parameter_display(params),
        "framing": RISK_FRAMING,
        "risk_level": level["key"],
        "risk_label": level["label"],
        "risk_copy": _risk_copy(level["key"]),
        "verdict": level["key"],
        "compliance_score": round(compliance_score, 1),
        "risk_score": round(compliance_score, 1),
        "risk_score_low": round(score_low, 1),
        "risk_score_high": round(score_high, 1),
        "risk_score_plus_minus": plus_minus,
        "confidence_score": round(confidence, 1),
        "confidence_label": _confidence_label(confidence),
        "confidence_ceiling": CONFIDENCE_CEILING,
        "confidence_note": CONFIDENCE_CEILING_NOTE,
        "bands": bands,
        "worst_band": bands[worst_index]["key"],
        "top_risk_factor": risk_factor,
        "shap": shap,
        "countermeasures": measures,
        "sensitivity": {
            "note": (
                "Change in risk score from a small nudge of each design parameter, "
                "evaluated through the monotone-constrained models. Negative "
                "delta_score means the nudge increased predicted EMC risk."
            ),
            "bars": sensitivity,
        },
        "signals": _serialize_signals(simulation),
        "power_quality": _serialize_power_quality(extracted),
        "spectrum": {
            "frequency_hz": [round(float(f), 1) for f in spectrum.frequency_hz],
            "emission_dbuv": [round(float(v), 2) for v in spectrum.emission_dbuv],
            "limit_dbuv": [round(float(v), 2) for v in spectrum.limit_dbuv],
        },
        "simulation_diagnostics": {
            key: round(float(value), 3)
            for key, value in extracted.diagnostics.items()
        },
        "model_info": {
            "artifact_version": models.artifact_version,
            "feature_count": len(FEATURE_NAMES),
            "design_feature_count": len(DESIGN_FEATURE_SPECS),
            "monotone_constraints": list(MONOTONE_CONSTRAINTS_RISK),
            "simulation_consistency": consistency,
            "ensemble_size": len(models.ensemble_regressors),
        },
        "limit_curve": {
            "anchors_hz_dbuv": [list(a) for a in LIMIT_CURVE_ANCHORS],
            "description": LIMIT_CURVE_DESCRIPTION,
            "is_synthetic": True,
        },
        "disclaimer": DISCLAIMER_SHORT,
        "disclaimer_long": DISCLAIMER_LONG,
        "assumptions_scope": SCOPE_STATEMENT,
        "radiated": radiated,
    }


def _parameter_display(params: DeviceParameters) -> List[Dict[str, Any]]:
    """Human-readable parameter rows for the results page and the PDF."""
    rows: List[Dict[str, Any]] = []
    for key in (
        "switching_frequency_khz", "dv_dt_v_per_us", "di_dt_a_per_us",
        "cable_length_m", "shielding_quality", "load_current_a",
        "input_filter_quality", "cm_choke_effectiveness", "clock_frequency_mhz",
    ):
        rge = PARAMETER_RANGE_BY_KEY[key]
        value = float(getattr(params, key))
        if key in ("shielding_quality", "input_filter_quality", "cm_choke_effectiveness"):
            # Stored 0-1, but shown as a percentage everywhere it is entered or
            # read, so the display must match rather than expose the raw fraction.
            formatted = f"{value * 100:.0f} %"
        else:
            formatted = f"{value:,.0f}" + (f" {rge.unit}" if rge.unit else "")
        rows.append({
            "key": key,
            "label": rge.label,
            "unit": rge.unit,
            "value": value,
            "formatted": formatted,
        })
    rows.append({
        "key": "pwm_modulation_type",
        "label": "PWM Modulation",
        "unit": "",
        "value": params.pwm_modulation_type,
        "formatted": PWM_MODULATION_LABELS[params.pwm_modulation_type],
    })
    rows.append({
        "key": "switching_device_type",
        "label": "Switching Device",
        "unit": "",
        "value": params.switching_device_type,
        "formatted": SWITCHING_DEVICE_LABELS[params.switching_device_type],
    })
    return rows


def _consistency_summary(models: ModelBundle) -> Dict[str, Any]:
    """Headline simulation-consistency figures, with the ablation alongside.

    Deliberately labelled "simulation consistency" everywhere, never "accuracy":
    these numbers describe agreement with our own physics model, not with measured
    hardware.
    """
    report = models.report or {}
    band_metrics = report.get("band_metrics", [])
    ablation = (report.get("design_only_ablation") or {}).get("band_metrics", [])

    def _mean(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(values)), 4) if values else None

    return {
        "note": (
            "Agreement between the models and this tool's own physics simulation "
            "on held-out synthetic designs. Not real-world accuracy: no measured "
            "hardware data is used anywhere in this pipeline."
        ),
        "n_test_designs": report.get("n_test"),
        "n_training_designs": report.get("n_samples"),
        "mean_balanced_accuracy": _mean(band_metrics, "classifier_balanced_accuracy"),
        "mean_margin_mae_db": _mean(band_metrics, "margin_mae_db"),
        "per_band": [
            {
                "band_label": m["band_label"],
                "balanced_accuracy": m["classifier_balanced_accuracy"],
                "roc_auc": m["classifier_roc_auc"],
                "margin_mae_db": m["margin_mae_db"],
                "margin_r2": m["margin_r2"],
            }
            for m in band_metrics
        ],
        "design_only_ablation": {
            "note": (
                "The same models restricted to the six design parameters, with all "
                "spectral features removed. The headline figures are high partly by "
                "construction, because the band peak level is nearly a sufficient "
                "statistic for the margin label; these show the harder problem of "
                "predicting compliance from the design alone."
            ),
            "mean_balanced_accuracy": _mean(
                ablation, "classifier_balanced_accuracy"
            ),
            "mean_margin_mae_db": _mean(ablation, "margin_mae_db"),
            "per_band": [
                {
                    "band_label": m["band_label"],
                    "balanced_accuracy": m["classifier_balanced_accuracy"],
                    "roc_auc": m["classifier_roc_auc"],
                    "margin_mae_db": m["margin_mae_db"],
                    "margin_r2": m["margin_r2"],
                }
                for m in ablation
            ],
        },
        "seed_stability": report.get("seed_stability"),
    }


def sensitivity_analysis(
    params: DeviceParameters,
    *,
    bundle: Optional[ModelBundle] = None,
) -> Dict[str, Any]:
    """Physics-level finite-difference of the risk score.

    Re-simulates the configuration with each numeric parameter nudged toward
    lower EMC risk (or, for input_filter_quality, toward more filtering) and
    reports the resulting change in risk score. Slower than the in-payload
    model sensitivity; intended for the tornado chart when the user wants the
    simulator's own gradient.
    """
    base = predict(params, bundle=bundle)
    base_score = float(base["risk_score"])
    bars: List[Dict[str, Any]] = []

    steps = {
        "switching_frequency_khz": ("mul", 0.85),
        "dv_dt_v_per_us": ("mul", 0.80),
        "cable_length_m": ("mul", 0.80),
        "shielding_quality": ("add", 0.10),
        "load_current_a": ("mul", 0.85),
        "input_filter_quality": ("add", 0.15),
    }
    for key, (mode, amount) in steps.items():
        current = float(getattr(params, key))
        rge = PARAMETER_RANGE_BY_KEY[key]
        trial = current * amount if mode == "mul" else current + amount
        trial = float(np.clip(trial, rge.minimum, rge.maximum))
        if abs(trial - current) < 1e-9:
            continue
        kwargs = params.as_dict()
        kwargs[key] = trial
        nudged = DeviceParameters.clamped(**kwargs)
        outcome = predict(nudged, bundle=bundle)
        bars.append({
            "parameter": key,
            "label": rge.label,
            "from_value": current,
            "to_value": trial,
            "delta_score": round(float(outcome["risk_score"]) - base_score, 2),
        })
    bars.sort(key=lambda row: abs(float(row["delta_score"])), reverse=True)
    return {
        "baseline_score": base_score,
        "bars": bars,
        "note": (
            "Each bar is a one-at-a-time physics re-simulation. A positive "
            "delta means the suggested nudge lowered predicted EMC risk."
        ),
    }


def compare_assessments(
    baseline: DeviceParameters,
    candidate: DeviceParameters,
    *,
    baseline_name: str = "Baseline",
    candidate_name: str = "Candidate",
    bundle: Optional[ModelBundle] = None,
) -> Dict[str, Any]:
    """Side-by-side assessment used by Compare Mode."""
    models = bundle or load_models()
    left = predict(baseline, device_name=baseline_name, bundle=models)
    right = predict(candidate, device_name=candidate_name, bundle=models)
    band_diff = []
    for a, b in zip(left["bands"], right["bands"]):
        band_diff.append({
            "key": a["key"],
            "label": a["label"],
            "baseline_margin_db": a["predicted_margin_db"],
            "candidate_margin_db": b["predicted_margin_db"],
            "delta_margin_db": round(
                b["predicted_margin_db"] - a["predicted_margin_db"], 2
            ),
        })
    return {
        "baseline": left,
        "candidate": right,
        "delta_risk_score": round(right["risk_score"] - left["risk_score"], 1),
        "band_diff": band_diff,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import json

    from devices import DEVICE_PROFILES

    models = load_models()
    print(f"{'device':30s} {'verdict':>8s} {'score':>7s} {'conf':>7s}  bands")
    for profile in DEVICE_PROFILES:
        result = predict(
            profile.parameters, device_id=profile.id, device_name=profile.name
        )
        flags = " ".join(
            f"{b['key'][-1].upper()}:{'P' if b['passes'] else 'F'}"
            f"({b['predicted_margin_db']:+.1f})"
            for b in result["bands"]
        )
        print(f"{profile.name:30s} {result['verdict']:>8s} "
              f"{result['compliance_score']:7.1f} "
              f"{result['confidence_score']:7.1f}  {flags}")
        print(f"{'':30s} risk: {result['top_risk_factor']['label']}")

    sample = predict(DEVICE_PROFILES[0].parameters)
    print("\nresult keys:", ", ".join(sorted(sample.keys())))
    print(f"payload size: {len(json.dumps(sample)) / 1024:.1f} kB")
