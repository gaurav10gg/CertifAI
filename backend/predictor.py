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
    DeviceParameters,
    simulate_device,
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "certifai_models.joblib"

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

# Confidence is capped below 100 on purpose.
#
# The figure computed below is the probability that the verdict is on the correct
# side of the limit *given the model's measured error against our own simulator*.
# That is only one of the two uncertainties in play. The other -- how well the
# synthetic physics model matches real hardware -- is currently unquantified,
# because there is no measured data anywhere in this pipeline (see calibrate.py).
#
# Since the unquantified term cannot be folded into the arithmetic, the reported
# figure is capped instead. Printing "100/100 confidence" for an assessment whose
# dominant error source has never been measured would be the single most
# misleading number this tool could produce.
CONFIDENCE_CEILING: float = 97.0

CONFIDENCE_CEILING_NOTE: str = (
    "Confidence is the probability that this verdict falls on the correct side of "
    "the limit line, given the models' measured margin error against our physics "
    "simulation. It is capped at "
    f"{CONFIDENCE_CEILING:.0f} because it does not include how closely that "
    "simulation matches real hardware, which is unquantified until measured "
    "calibration data exists."
)

# Applied once per band where the classifier and the margin regressor disagree
# about which side of the limit the design falls on.
DISAGREEMENT_PENALTY: float = 0.80

DISCLAIMER_SHORT: str = (
    "This is a simulation-based prediction validated against our physics model, "
    "not a substitute for accredited EMC lab certification."
)

DISCLAIMER_LONG: str = (
    "This is a simulation-based pre-compliance assessment intended to support "
    "early-stage design decisions. It does not replace accredited EMC "
    "certification testing (e.g. per EN 12016)."
)


class ModelNotTrainedError(RuntimeError):
    """Raised when the saved model artifacts are missing."""


@dataclass(frozen=True)
class ModelBundle:
    """Loaded artifacts plus the metadata needed to interpret them."""

    scaler: Any
    classifiers: Sequence[xgb.XGBClassifier]
    regressors: Sequence[xgb.XGBRegressor]
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
        "Improve the screen: 360-degree gland terminations at both ends plus a "
        "common-mode choke on the motor leads, targeting an effectiveness of about "
        "{target:.2f}. This is normally the cheapest large improvement."
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
    if key == "shielding_quality":
        target = current + _SHIELDING_STEP
    else:
        target = current * _MULTIPLICATIVE_STEP[key]
    return float(np.clip(target, rge.minimum, rge.maximum))


def _shap_contributions(
    classifier: xgb.XGBClassifier, scaled_features: np.ndarray
) -> np.ndarray:
    """Exact per-feature SHAP contributions to the log-odds of failure.

    XGBoost computes these analytically for tree ensembles (``pred_contribs``), so
    the attribution is exact for this model rather than an approximation. Falls
    back to gain-based importance if the booster cannot provide contributions.
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
    all_pass = all(b["passes"] for b in bands)

    # The band with the least predicted headroom drives the narrative, whether or
    # not it actually fails.
    worst_index = int(np.argmin(predicted_margins))
    risk_factor = _risk_factor(
        models, extracted.feature_vector, scaled, params,
        worst_index, not bands[worst_index]["passes"],
    )

    spectrum = extracted.spectrum
    consistency = _consistency_summary(models)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device_id": device_id,
        "device_name": device_name or "Custom Configuration",
        "parameters": params.as_dict(),
        "parameter_display": _parameter_display(params),
        "verdict": "PASS" if all_pass else "FAIL",
        "compliance_score": round(compliance_score, 1),
        "confidence_score": round(confidence, 1),
        "confidence_label": _confidence_label(confidence),
        "confidence_ceiling": CONFIDENCE_CEILING,
        "confidence_note": CONFIDENCE_CEILING_NOTE,
        "bands": bands,
        "worst_band": bands[worst_index]["key"],
        "top_risk_factor": risk_factor,
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
            "monotone_constraints": list(MONOTONE_CONSTRAINTS_RISK),
            "simulation_consistency": consistency,
        },
        "limit_curve": {
            "anchors_hz_dbuv": [list(a) for a in LIMIT_CURVE_ANCHORS],
            "description": LIMIT_CURVE_DESCRIPTION,
            "is_synthetic": True,
        },
        "disclaimer": DISCLAIMER_SHORT,
        "disclaimer_long": DISCLAIMER_LONG,
    }


def _parameter_display(params: DeviceParameters) -> List[Dict[str, Any]]:
    """Human-readable parameter rows for the results page and the PDF."""
    rows: List[Dict[str, Any]] = []
    for key in ("switching_frequency_khz", "dv_dt_v_per_us", "cable_length_m",
                "shielding_quality", "load_current_a"):
        rge = PARAMETER_RANGE_BY_KEY[key]
        value = float(getattr(params, key))
        if key == "shielding_quality":
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
