"""Phase 7 validation suite: monotonicity sweeps and SHAP additivity.

These checks exercise the *trained artifacts*. They are not chamber accuracy.

  * Monotonicity: 200 random 18-feature vectors, each of the six constrained
    design parameters swept while the rest are held fixed. The risk score
    (higher = more headroom) must move in the physics-legal direction.
  * SHAP additivity: exact tree ``pred_contribs`` on the design-only margin
    models. ``sum(SHAP) + bias`` must recover the predicted margin.

Run from backend/::

    python tools/run_validation.py

The JSON at ``models/validation_suite.json`` is what ``GET /api/validation``
serves, so the Model Validation page shows measured numbers rather than copy.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from features import DESIGN_FEATURE_SPECS, EMC_BANDS, FEATURE_NAMES
from predictor import MODEL_PATH, _compliance_score, load_models
from simulate import PARAMETER_RANGE_BY_KEY

SUITE_PATH = MODEL_PATH.parent / "validation_suite.json"

N_MONO_CONFIGS = 200
N_MONO_STEPS = 8
N_SHAP_DESIGNS = 200
MONO_SEED = 20260916
SHAP_SEED = 20260917
ILLEGAL_SCORE_TOL = 1e-6
SHAP_PASS_MEAN_ABS = 1e-4


def _feature_label(name: str) -> str:
    if name == "pwm_cm_penalty_db":
        return "PWM modulation"
    return PARAMETER_RANGE_BY_KEY[name].label


def _sci(value: float) -> str:
    """Compact scientific notation: 2.4e-6 rather than 2.4e-06."""
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def _risk_scores(bundle, vectors: np.ndarray) -> np.ndarray:
    """Risk score for each row of an (n, 18) feature matrix."""
    scaled = bundle.scaler.transform(vectors)
    margins = np.column_stack(
        [np.asarray(reg.predict(scaled), dtype=float) for reg in bundle.regressors]
    )
    return np.array([_compliance_score(row) for row in margins], dtype=float)


def check_monotonicity(
    bundle,
    *,
    n_configs: int = N_MONO_CONFIGS,
    n_steps: int = N_MONO_STEPS,
    seed: int = MONO_SEED,
) -> Dict[str, Any]:
    """Finite-difference sweep of each constrained design parameter."""
    rng = np.random.default_rng(seed)
    mean = np.asarray(bundle.scaler.mean_, dtype=float)
    scale = np.asarray(bundle.scaler.scale_, dtype=float)
    n_features = len(DESIGN_FEATURE_SPECS)

    bases = mean + rng.normal(0.0, 1.0, size=(n_configs, mean.shape[0])) * scale
    # One stacked matrix: config-major, then feature, then sweep step.
    stacked = np.repeat(bases, n_features * n_steps, axis=0)
    grids = []
    for spec in DESIGN_FEATURE_SPECS:
        index = FEATURE_NAMES.index(spec.name)
        lo = mean[index] - 2.0 * scale[index]
        hi = mean[index] + 2.0 * scale[index]
        grids.append((spec, index, np.linspace(lo, hi, n_steps)))

    row = 0
    for _config in range(n_configs):
        for _spec, index, grid in grids:
            for value in grid:
                stacked[row, index] = value
                row += 1

    scores = _risk_scores(bundle, stacked).reshape(n_configs, n_features, n_steps)

    failures: List[Dict[str, Any]] = []
    per_feature = {
        spec.name: {
            "label": _feature_label(spec.name),
            "checked": 0,
            "violations": 0,
            "risk_sign": spec.risk_sign,
        }
        for spec in DESIGN_FEATURE_SPECS
    }

    for config_index in range(n_configs):
        for feature_index, spec in enumerate(DESIGN_FEATURE_SPECS):
            sweep = scores[config_index, feature_index]
            diffs = np.diff(sweep)
            illegal = diffs * spec.risk_sign > ILLEGAL_SCORE_TOL
            per_feature[spec.name]["checked"] += 1
            if np.any(illegal):
                per_feature[spec.name]["violations"] += 1
                failures.append({
                    "feature": spec.name,
                    "label": _feature_label(spec.name),
                    "risk_sign": spec.risk_sign,
                    "scores": [round(float(s), 4) for s in sweep],
                    "max_illegal_delta": round(
                        float(np.max(diffs * spec.risk_sign)), 4
                    ),
                })

    n_checks = n_configs * len(DESIGN_FEATURE_SPECS)
    n_fail = len(failures)
    passed = n_fail == 0
    consistent_pct = 100.0 * (n_checks - n_fail) / n_checks if n_checks else 0.0
    if passed:
        headline = (
            f"Monotonicity verified across {n_configs} randomized configurations: "
            f"{consistent_pct:.0f}% consistent."
        )
    else:
        headline = (
            f"Monotonicity failed on {n_fail} of {n_checks} sweeps across "
            f"{n_configs} randomized configurations."
        )
    return {
        "passed": passed,
        "headline": headline,
        "n_configs": n_configs,
        "n_steps": n_steps,
        "n_checks": n_checks,
        "n_violations": n_fail,
        "consistent_percent": round(consistent_pct, 4),
        "per_feature": per_feature,
        "first_failures": failures[:8],
        "note": (
            "Model-space finite differences with spectral features held fixed. "
            "A pass means the trained trees respect the physics constraint vector "
            "on this sample. It is not a hardware validation."
        ),
    }


def check_shap_additivity(
    bundle,
    *,
    n_designs: int = N_SHAP_DESIGNS,
    seed: int = SHAP_SEED,
) -> Dict[str, Any]:
    """``sum(SHAP) + bias`` must recover the design-only predicted margin."""
    if not bundle.design_only_regressors or bundle.design_only_scaler is None:
        raise RuntimeError(
            "Design-only margin models are missing. "
            "Run: python train_model.py --design-only"
        )

    import xgboost as xgb

    rng = np.random.default_rng(seed)
    mean = np.asarray(bundle.scaler.mean_, dtype=float)
    scale = np.asarray(bundle.scaler.scale_, dtype=float)
    raw = mean + rng.normal(0.0, 1.0, size=(n_designs, mean.shape[0])) * scale
    names = list(bundle.design_only_feature_names) or [
        spec.name for spec in DESIGN_FEATURE_SPECS
    ]
    indices = [FEATURE_NAMES.index(name) for name in names]
    scaled = bundle.design_only_scaler.transform(raw[:, indices])
    matrix = xgb.DMatrix(scaled)

    residuals: List[float] = []
    per_band: List[Dict[str, Any]] = []
    for band_index, regressor in enumerate(bundle.design_only_regressors):
        booster = regressor.get_booster()
        kwargs: Dict[str, Any] = {"validate_features": False}
        best = getattr(regressor, "best_iteration", None)
        if best is not None:
            kwargs["iteration_range"] = (0, int(best) + 1)
        predicted = np.asarray(booster.predict(matrix, **kwargs), dtype=float)
        contrib = np.asarray(
            booster.predict(matrix, pred_contribs=True, **kwargs), dtype=float
        )
        shap_sum = contrib[:, :-1].sum(axis=1)
        bias = contrib[:, -1]
        band_residuals = (shap_sum + bias) - predicted
        residuals.extend(float(v) for v in band_residuals)
        abs_band = np.abs(band_residuals)
        per_band.append({
            "band": EMC_BANDS[band_index].label,
            "n": n_designs,
            "mean_abs_error_db": float(np.mean(abs_band)),
            "max_abs_error_db": float(np.max(abs_band)),
        })

    abs_all = np.abs(residuals)
    mean_abs = float(np.mean(abs_all))
    max_abs = float(np.max(abs_all))
    passed = mean_abs < SHAP_PASS_MEAN_ABS
    headline = f"SHAP additivity verified: mean error {_sci(mean_abs)}."
    if not passed:
        headline = (
            f"SHAP additivity failed: mean error {_sci(mean_abs)} "
            f"(threshold {_sci(SHAP_PASS_MEAN_ABS)})."
        )
    return {
        "passed": passed,
        "headline": headline,
        "n_designs": n_designs,
        "n_checks": len(residuals),
        "mean_abs_error_db": mean_abs,
        "max_abs_error_db": max_abs,
        "mean_abs_error_display": _sci(mean_abs),
        "max_abs_error_display": _sci(max_abs),
        "per_band": per_band,
        "note": (
            "Exact tree SHAP (XGBoost pred_contribs) on the design-only margin "
            "regressors. The identity is sum(SHAP) + bias ≈ predicted margin (dB). "
            "This is a check that the waterfall is an accounting of the model, "
            "not a hardware claim."
        ),
    }


def run_suite(
    *,
    n_mono_configs: int = N_MONO_CONFIGS,
    n_shap_designs: int = N_SHAP_DESIGNS,
    bundle=None,
) -> Dict[str, Any]:
    """Run both checks against the loaded artifacts."""
    models = bundle or load_models()
    monotonicity = check_monotonicity(models, n_configs=n_mono_configs)
    shap = check_shap_additivity(models, n_designs=n_shap_designs)
    report = models.report or {}
    ablation = report.get("design_only_ablation") or {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact_version": models.artifact_version,
        "passed": bool(monotonicity["passed"] and shap["passed"]),
        "headlines": {
            "monotonicity": monotonicity["headline"],
            "shap_additivity": shap["headline"],
        },
        "monotonicity": monotonicity,
        "shap_additivity": shap,
        "simulation_consistency": {
            "metric_semantics": report.get("metric_semantics"),
            "n_test": report.get("n_test"),
            "design_only_ablation": ablation,
        },
        "note": (
            "These checks prove the deployed models honour the physics constraint "
            "vector and that SHAP contributions add up to the prediction. They "
            "measure agreement with this tool's own simulator and with itself, "
            "not agreement with accredited lab measurements."
        ),
    }


def write_suite(report: Dict[str, Any], path: Optional[Path] = None) -> Path:
    destination = path or SUITE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    return destination


def load_suite(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    destination = path or SUITE_PATH
    if not destination.exists():
        return None
    return json.loads(destination.read_text(encoding="utf-8"))
