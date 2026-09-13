"""
train_model.py -- Build the physics-informed EMC compliance models.

Pipeline
--------
    1. Draw N random designs from the admissible parameter space.
    2. Push each through simulate.py -> features.py.
    3. Label each band: fail flag (margin <= 0) and continuous margin in dB.
    4. Fit, per band:
         * XGBClassifier  -> P(band fails)
         * XGBRegressor   -> margin-to-limit in dB
       both with physics-informed ``monotone_constraints``.
    5. Validate on a held-out split and write everything to models/.

What "physics-informed" means here
----------------------------------
Every one of the 18 model inputs has a *known* sign of influence on failure risk,
derived from the emission physics rather than from the data (see
``features.FEATURE_SPECS`` for the per-feature rationale). Those signs are passed
to XGBoost as ``monotone_constraints``, so the fitted trees are structurally
incapable of learning a physically impossible relationship -- for example, they
cannot conclude that adding cable screening raises emissions, no matter what a
noisy corner of the training set happens to look like.

Concretely, the constraint vector is (in feature order):

    switching_frequency_khz   +1      band_{a,b,c}_peak_dbuv        +1
    dv_dt_v_per_us            +1      band_{a,b,c}_rms_dbuv         +1
    cable_length_m            +1      band_{a,b,c}_harmonic_count   +1
    shielding_quality         -1      band_{a,b,c}_thd_score        +1
    load_current_a            +1
    pwm_cm_penalty_db         +1

Signs are stated with respect to *failure risk*. The margin regressors predict
headroom (higher = safer), so they receive the negated vector.

Why an ML surrogate at all, when the simulator is right there?
-------------------------------------------------------------
  * Speed, via the design-only ablation described below. One simulation takes
    ~180 ms; the fitted models answer in well under a millisecond. That is what
    makes interactive parameter sweeps and design-space exploration practical,
    and the ablation shows it costs only ~1-2 dB of margin error to skip the
    simulation entirely.
  * A calibration surface. The models can be corrected towards real measured data
    without touching the simulator; see ``calibrate.py``.

Note what is *not* on that list. An earlier version of this file claimed the
surrogate usefully suppresses simulator noise. ``audit_seed_stability`` was added
to check that, and it does not hold: re-simulating one design under different
noise seeds moves the margin by only ~0.25 dB, and the surrogate's spread is the
same to within measurement error. The simulator is already reproducible, so there
is no noise for the model to reject. The audit is kept in the report because a
measured negative result is worth more than a plausible-sounding claim.

Honesty note on metrics
-----------------------
Every accuracy figure this script produces is a **simulation-consistency score**:
it measures how faithfully the models reproduce *our own physics simulation* on
designs they were not trained on. It is NOT a real-world accuracy, because no
measured hardware data is involved anywhere in this pipeline. The reports, the
API and the UI all use the "simulation-consistency" wording for this reason.

Expect these scores to come out very high (R^2 > 0.99, AUC > 0.99). That is not a
sign of a well-generalising model and should not be advertised as one. The reason
is structural: the label for a band is ``limit(f) - emission(f)`` minimised over
the band, and ``band_*_peak_dbuv`` is the maximum of that same emission trace, so
the peak feature is very nearly a sufficient statistic for the label. The two
differ only because the limit line is not flat within bands A and C, so the worst
-margin frequency need not be the peak frequency.

Because a headline number that is trivially high is worth little, ``train()``
also fits a **design-only ablation**: the same models restricted to the six
design parameters, with no spectral features at all. That variant has to infer
the spectrum from the design, which is the genuinely hard version of the problem
and is also what you would deploy if you wanted a compliance estimate *without*
running the simulation. Both sets of figures go into the report, and the UI shows
the ablation alongside the headline so the comparison is not buried.

Usage
-----
    python train_model.py                 # 5000 samples, all cores
    python train_model.py --n-samples 800 --jobs 4
    python train_model.py --report-only   # re-print the saved validation report
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import (
    DESIGN_FEATURE_SPECS,
    EMC_BANDS,
    FEATURE_NAMES,
    LIMIT_CURVE_ANCHORS,
    LIMIT_CURVE_DESCRIPTION,
    MONOTONE_CONSTRAINTS_MARGIN,
    MONOTONE_CONSTRAINTS_RISK,
    band_margins,
    extract_features,
    xgboost_monotone_string,
)
from simulate import (
    COUPLING_CALIBRATION,
    DeviceParameters,
    N_SEGMENTS,
    SAMPLE_RATE_HZ,
    sample_random_parameters,
    simulate_device,
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "certifai_models.joblib"
REPORT_PATH = MODEL_DIR / "validation_report.json"
DATASET_PATH = MODEL_DIR / "training_dataset.npz"

ARTIFACT_VERSION = "1.0.0"

# Fraction of the dataset held out for validation, then for the final test split.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_STATE = 20260912

CLASSIFIER_PARAMS: Dict[str, object] = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=4,
    reg_lambda=1.5,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    early_stopping_rounds=40,
)

REGRESSOR_PARAMS: Dict[str, object] = dict(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=4,
    reg_lambda=1.5,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    early_stopping_rounds=40,
)


# ---------------------------------------------------------------------------
# 1-3. Synthetic dataset generation
# ---------------------------------------------------------------------------
def _evaluate_design(params: DeviceParameters, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run one design through the full physics chain. Returns (features, margins)."""
    bundle = extract_features(simulate_device(params, seed=seed))
    return bundle.feature_vector, band_margins(bundle)


def generate_dataset(
    n_samples: int, jobs: int, seed: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, List[DeviceParameters]]:
    """Simulate ``n_samples`` random designs and extract their features/labels.

    Each design gets its own explicit seed so the dataset is fully reproducible
    and so a design can be re-simulated later for debugging.
    """
    rng = np.random.default_rng(seed)
    designs = [sample_random_parameters(rng) for _ in range(n_samples)]
    seeds = rng.integers(0, 2 ** 31 - 1, size=n_samples).tolist()

    print(f"Simulating {n_samples} designs on {jobs} worker(s)...")
    started = time.time()
    results = joblib.Parallel(n_jobs=jobs, verbose=1, batch_size=8)(
        joblib.delayed(_evaluate_design)(design, int(s))
        for design, s in zip(designs, seeds)
    )
    elapsed = time.time() - started

    features = np.vstack([r[0] for r in results])
    margins = np.vstack([r[1] for r in results])
    print(
        f"  done in {elapsed:.1f}s "
        f"({elapsed / n_samples * 1e3:.0f} ms/design wall-clock)"
    )
    return features, margins, designs


# ---------------------------------------------------------------------------
# 4. Model fitting
# ---------------------------------------------------------------------------
@dataclass
class BandMetrics:
    """Simulation-consistency metrics for one band. NOT real-world accuracy."""

    band_key: str
    band_label: str
    fail_rate: float
    classifier_accuracy: float
    classifier_balanced_accuracy: float
    classifier_roc_auc: Optional[float]
    margin_mae_db: float
    margin_rmse_db: float
    margin_r2: float
    classifier_best_iteration: int
    regressor_best_iteration: int


def _fit_band_models(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_fail_train: np.ndarray,
    y_fail_val: np.ndarray,
    y_margin_train: np.ndarray,
    y_margin_val: np.ndarray,
) -> Tuple[xgb.XGBClassifier, xgb.XGBRegressor]:
    """Fit the fail classifier and margin regressor for a single band."""
    # Failure is the minority class in the high-frequency band; weight it up so
    # the classifier does not simply predict "compliant" everywhere.
    n_fail = max(int(y_fail_train.sum()), 1)
    scale_pos_weight = float((len(y_fail_train) - n_fail) / n_fail)

    classifier = xgb.XGBClassifier(
        **CLASSIFIER_PARAMS,
        monotone_constraints=xgboost_monotone_string(MONOTONE_CONSTRAINTS_RISK),
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )
    classifier.fit(
        x_train, y_fail_train, eval_set=[(x_val, y_fail_val)], verbose=False
    )

    # Headroom (higher = safer) is the opposite sense to failure risk, so the
    # constraint vector is negated.
    regressor = xgb.XGBRegressor(
        **REGRESSOR_PARAMS,
        monotone_constraints=xgboost_monotone_string(MONOTONE_CONSTRAINTS_MARGIN),
        random_state=RANDOM_STATE,
    )
    regressor.fit(
        x_train, y_margin_train, eval_set=[(x_val, y_margin_val)], verbose=False
    )

    return classifier, regressor


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------
def _band_metrics(
    band_index: int,
    classifier: xgb.XGBClassifier,
    regressor: xgb.XGBRegressor,
    x_test: np.ndarray,
    y_fail_test: np.ndarray,
    y_margin_test: np.ndarray,
) -> BandMetrics:
    band = EMC_BANDS[band_index]
    p_fail = classifier.predict_proba(x_test)[:, 1]
    predicted_fail = (p_fail >= 0.5).astype(int)
    predicted_margin = regressor.predict(x_test)

    both_classes = len(np.unique(y_fail_test)) > 1
    residual = predicted_margin - y_margin_test

    return BandMetrics(
        band_key=band.key,
        band_label=band.label,
        fail_rate=float(y_fail_test.mean()),
        classifier_accuracy=float((predicted_fail == y_fail_test).mean()),
        classifier_balanced_accuracy=(
            float(balanced_accuracy_score(y_fail_test, predicted_fail))
            if both_classes else float("nan")
        ),
        classifier_roc_auc=(
            float(roc_auc_score(y_fail_test, p_fail)) if both_classes else None
        ),
        margin_mae_db=float(mean_absolute_error(y_margin_test, predicted_margin)),
        margin_rmse_db=float(np.sqrt(np.mean(residual ** 2))),
        margin_r2=float(r2_score(y_margin_test, predicted_margin)),
        classifier_best_iteration=int(getattr(classifier, "best_iteration", -1) or -1),
        regressor_best_iteration=int(getattr(regressor, "best_iteration", -1) or -1),
    )


def audit_model_monotonicity(
    classifiers: Sequence[xgb.XGBClassifier],
    scaler: StandardScaler,
    x_raw_test: np.ndarray,
    n_probe: int = 400,
) -> List[Dict[str, object]]:
    """Confirm the fitted models actually honour the physics constraints.

    For each design feature we nudge it towards its riskier end on held-out
    designs and check that no band's predicted failure probability moves the wrong
    way. With ``monotone_constraints`` in force this must hold exactly, so any
    violation here means the constraint vector and the feature order have drifted
    apart -- a class of bug that is otherwise silent.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    subset = x_raw_test[rng.choice(len(x_raw_test), min(n_probe, len(x_raw_test)),
                                   replace=False)]
    findings: List[Dict[str, object]] = []

    for spec in DESIGN_FEATURE_SPECS:
        index = FEATURE_NAMES.index(spec.name)
        spread = float(np.std(x_raw_test[:, index])) or 1.0

        nudged = subset.copy()
        # Move the feature in the direction that must increase risk.
        nudged[:, index] += spec.risk_sign * 0.25 * spread

        base = scaler.transform(subset)
        after = scaler.transform(nudged)

        worst_drop = 0.0
        for classifier in classifiers:
            delta = (
                classifier.predict_proba(after)[:, 1]
                - classifier.predict_proba(base)[:, 1]
            )
            worst_drop = min(worst_drop, float(delta.min()))

        findings.append({
            "feature": spec.name,
            "risk_sign": spec.risk_sign,
            "worst_risk_decrease": worst_drop,
            "respects_constraint": bool(worst_drop >= -1e-9),
        })

    return findings


def design_only_ablation(
    x_raw: np.ndarray,
    margins: np.ndarray,
    y_fail: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> List[Dict[str, object]]:
    """Refit using ONLY the six design parameters, as an honesty benchmark.

    The headline models see spectral features extracted from the simulated
    waveform, and ``band_*_peak_dbuv`` is very nearly a sufficient statistic for
    the margin label -- so their scores are high almost by construction. This
    ablation removes every spectral feature, forcing the models to infer the
    emission spectrum from the design alone. It is the harder and more meaningful
    problem, and its numbers are the ones to quote when asked "how well does this
    actually predict anything?".
    """
    indices = [FEATURE_NAMES.index(spec.name) for spec in DESIGN_FEATURE_SPECS]
    constraints_risk = tuple(spec.risk_sign for spec in DESIGN_FEATURE_SPECS)
    constraints_margin = tuple(-c for c in constraints_risk)

    x_design = x_raw[:, indices]
    scaler = StandardScaler().fit(x_design[idx_train])
    x_train = scaler.transform(x_design[idx_train])
    x_val = scaler.transform(x_design[idx_val])
    x_test = scaler.transform(x_design[idx_test])

    results: List[Dict[str, object]] = []
    for index, band in enumerate(EMC_BANDS):
        n_fail = max(int(y_fail[idx_train, index].sum()), 1)
        classifier = xgb.XGBClassifier(
            **CLASSIFIER_PARAMS,
            monotone_constraints=xgboost_monotone_string(constraints_risk),
            scale_pos_weight=float((len(idx_train) - n_fail) / n_fail),
            random_state=RANDOM_STATE,
        )
        classifier.fit(
            x_train, y_fail[idx_train, index],
            eval_set=[(x_val, y_fail[idx_val, index])], verbose=False,
        )
        regressor = xgb.XGBRegressor(
            **REGRESSOR_PARAMS,
            monotone_constraints=xgboost_monotone_string(constraints_margin),
            random_state=RANDOM_STATE,
        )
        regressor.fit(
            x_train, margins[idx_train, index],
            eval_set=[(x_val, margins[idx_val, index])], verbose=False,
        )

        metrics = _band_metrics(
            index, classifier, regressor, x_test,
            y_fail[idx_test, index], margins[idx_test, index],
        )
        results.append(asdict(metrics))

    return results


def audit_seed_stability(
    regressors: Sequence[xgb.XGBRegressor],
    scaler: StandardScaler,
    n_designs: int = 24,
    n_seeds: int = 5,
) -> Dict[str, object]:
    """Measure how much run-to-run simulator noise the surrogate absorbs.

    The same design is simulated under several noise seeds; we compare the spread
    of the raw simulator margin against the spread of the model's predicted margin.

    Measured outcome: both are around 0.25 dB, i.e. the surrogate provides no
    meaningful variance reduction, because the simulator is already reproducible to
    well under a dB. This audit exists to keep that claim honest rather than to
    support it -- see the module docstring.
    """
    rng = np.random.default_rng(RANDOM_STATE + 1)
    simulator_sd: List[float] = []
    model_sd: List[float] = []

    for _ in range(n_designs):
        design = sample_random_parameters(rng)
        sim_margins, pred_margins = [], []
        for seed in range(n_seeds):
            bundle = extract_features(simulate_device(design, seed=seed))
            sim_margins.append(band_margins(bundle))
            scaled = scaler.transform(bundle.feature_vector.reshape(1, -1))
            pred_margins.append([float(r.predict(scaled)[0]) for r in regressors])
        simulator_sd.append(np.std(np.array(sim_margins), axis=0).mean())
        model_sd.append(np.std(np.array(pred_margins), axis=0).mean())

    sim_mean = float(np.mean(simulator_sd))
    mdl_mean = float(np.mean(model_sd))
    return {
        "n_designs": n_designs,
        "n_seeds_per_design": n_seeds,
        "simulator_margin_sd_db": sim_mean,
        "model_margin_sd_db": mdl_mean,
        "variance_reduction_factor": (sim_mean / mdl_mean) if mdl_mean > 0 else None,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def train(n_samples: int, jobs: int, skip_stability: bool = False) -> Dict[str, object]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    x_raw, margins, designs = generate_dataset(n_samples, jobs)
    y_fail = (margins <= 0.0).astype(int)

    print("\nLabel balance (share of designs failing each band):")
    for index, band in enumerate(EMC_BANDS):
        print(f"  {band.label:18s} {y_fail[:, index].mean():6.1%}")
    print(f"  {'any band':18s} {y_fail.any(axis=1).mean():6.1%}")

    # Split once, and reuse the same indices for every band so the models are
    # directly comparable and the test set is never touched during fitting.
    idx = np.arange(n_samples)
    idx_fit, idx_test = train_test_split(
        idx, test_size=TEST_FRACTION, random_state=RANDOM_STATE,
        stratify=y_fail.any(axis=1),
    )
    idx_train, idx_val = train_test_split(
        idx_fit, test_size=VAL_FRACTION / (1.0 - TEST_FRACTION),
        random_state=RANDOM_STATE, stratify=y_fail[idx_fit].any(axis=1),
    )

    # StandardScaler is a strictly increasing affine map per feature, so it
    # preserves every monotone constraint. It is fitted here (and saved) so the
    # feature space is well conditioned for future calibration work, which may
    # use models that are not scale-invariant.
    scaler = StandardScaler().fit(x_raw[idx_train])
    x_train = scaler.transform(x_raw[idx_train])
    x_val = scaler.transform(x_raw[idx_val])
    x_test = scaler.transform(x_raw[idx_test])

    print(
        f"\nSplits: train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}"
    )
    print(f"Monotone constraints (risk)  : "
          f"{xgboost_monotone_string(MONOTONE_CONSTRAINTS_RISK)}")
    print(f"Monotone constraints (margin): "
          f"{xgboost_monotone_string(MONOTONE_CONSTRAINTS_MARGIN)}")

    classifiers: List[xgb.XGBClassifier] = []
    regressors: List[xgb.XGBRegressor] = []
    metrics: List[BandMetrics] = []

    for index, band in enumerate(EMC_BANDS):
        print(f"\nFitting {band.label} ...")
        classifier, regressor = _fit_band_models(
            x_train, x_val,
            y_fail[idx_train, index], y_fail[idx_val, index],
            margins[idx_train, index], margins[idx_val, index],
        )
        classifiers.append(classifier)
        regressors.append(regressor)

        band_metrics = _band_metrics(
            index, classifier, regressor, x_test,
            y_fail[idx_test, index], margins[idx_test, index],
        )
        metrics.append(band_metrics)
        auc = band_metrics.classifier_roc_auc
        print(
            f"  simulation-consistency: acc={band_metrics.classifier_accuracy:.3f} "
            f"balanced={band_metrics.classifier_balanced_accuracy:.3f} "
            f"auc={auc:.3f}" if auc is not None else
            f"  simulation-consistency: acc={band_metrics.classifier_accuracy:.3f}"
        )
        print(
            f"  margin regression     : MAE={band_metrics.margin_mae_db:.2f} dB "
            f"RMSE={band_metrics.margin_rmse_db:.2f} dB "
            f"R2={band_metrics.margin_r2:.4f}"
        )

    print("\nAuditing enforced monotonicity ...")
    monotonicity = audit_model_monotonicity(classifiers, scaler, x_raw[idx_test])
    for finding in monotonicity:
        status = "OK  " if finding["respects_constraint"] else "FAIL"
        print(f"  {status} {finding['feature']:26s} "
              f"worst risk decrease {finding['worst_risk_decrease']:+.2e}")

    print("\nFitting design-only ablation (6 features, no spectral inputs) ...")
    ablation = design_only_ablation(
        x_raw, margins, y_fail, idx_train, idx_val, idx_test
    )
    for m in ablation:
        auc = m["classifier_roc_auc"]
        print(f"  {m['band_label']:18s} "
              f"acc={m['classifier_accuracy']:.3f} "
              f"auc={(f'{auc:.3f}' if auc is not None else 'n/a')} "
              f"MAE={m['margin_mae_db']:.2f} dB R2={m['margin_r2']:.4f}")

    stability: Dict[str, object] = {}
    if not skip_stability:
        print("\nMeasuring seed stability (simulator vs surrogate) ...")
        stability = audit_seed_stability(regressors, scaler)
        print(f"  simulator margin SD : "
              f"{stability['simulator_margin_sd_db']:.2f} dB")
        print(f"  surrogate margin SD : "
              f"{stability['model_margin_sd_db']:.2f} dB")
        if stability.get("variance_reduction_factor"):
            print(f"  variance reduction  : "
                  f"{stability['variance_reduction_factor']:.1f}x")

    report = {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric_semantics": (
            "All accuracy figures are SIMULATION-CONSISTENCY scores: they measure "
            "agreement between the models and the physics simulation in "
            "simulate.py/features.py on held-out synthetic designs. They are NOT "
            "real-world accuracy against measured hardware, and must never be "
            "presented as such."
        ),
        "n_samples": n_samples,
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
        "feature_names": list(FEATURE_NAMES),
        "monotone_constraints_risk": list(MONOTONE_CONSTRAINTS_RISK),
        "monotone_constraints_margin": list(MONOTONE_CONSTRAINTS_MARGIN),
        "band_metrics": [asdict(m) for m in metrics],
        "design_only_ablation": {
            "note": (
                "Same models restricted to the six design parameters, with every "
                "spectral feature removed. The headline models see "
                "band_*_peak_dbuv, which is very nearly a sufficient statistic for "
                "the margin label, so their scores are high by construction. These "
                "ablation figures reflect the harder problem of predicting "
                "compliance from the design alone."
            ),
            "feature_names": [spec.name for spec in DESIGN_FEATURE_SPECS],
            "band_metrics": ablation,
        },
        "monotonicity_audit": monotonicity,
        "seed_stability": stability,
        "simulator_config": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "n_segments": N_SEGMENTS,
            "coupling_calibration": COUPLING_CALIBRATION,
        },
        "limit_curve": {
            "anchors_hz_dbuv": [list(a) for a in LIMIT_CURVE_ANCHORS],
            "description": LIMIT_CURVE_DESCRIPTION,
            "status": "SYNTHETIC -- not the normative EN 12016 limit table",
        },
        "environment": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
        },
    }

    bundle = {
        "artifact_version": ARTIFACT_VERSION,
        "scaler": scaler,
        "classifiers": classifiers,
        "regressors": regressors,
        "feature_names": list(FEATURE_NAMES),
        "band_keys": [b.key for b in EMC_BANDS],
        "monotone_constraints_risk": list(MONOTONE_CONSTRAINTS_RISK),
        "monotone_constraints_margin": list(MONOTONE_CONSTRAINTS_MARGIN),
        # Per-band held-out margin RMSE, used at inference time to turn a point
        # prediction into an honest confidence figure.
        "margin_rmse_db": [m.margin_rmse_db for m in metrics],
        "report": report,
    }

    joblib.dump(bundle, MODEL_PATH, compress=3)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        DATASET_PATH,
        features=x_raw,
        margins=margins,
        feature_names=np.array(FEATURE_NAMES),
        parameters=np.array([json.dumps(d.as_dict()) for d in designs]),
    )

    print(f"\nSaved models   -> {MODEL_PATH}")
    print(f"Saved report   -> {REPORT_PATH}")
    print(f"Saved dataset  -> {DATASET_PATH}")
    return report


def print_report(report: Dict[str, object]) -> None:
    print("\n" + "=" * 74)
    print("VALIDATION REPORT -- simulation-consistency, NOT real-world accuracy")
    print("=" * 74)
    print(f"generated        : {report['generated_at']}")
    print(f"samples          : {report['n_samples']} "
          f"(train {report['n_train']} / val {report['n_val']} / test {report['n_test']})")
    print()
    header = (f"{'band':18s} {'fail%':>6s} {'acc':>6s} {'bal-acc':>8s} "
              f"{'auc':>6s} {'MAE dB':>7s} {'RMSE dB':>8s} {'R2':>7s}")
    print(header)
    print("-" * len(header))
    for m in report["band_metrics"]:  # type: ignore[index]
        auc = m["classifier_roc_auc"]
        print(f"{m['band_label']:18s} {m['fail_rate']:6.1%} "
              f"{m['classifier_accuracy']:6.3f} "
              f"{m['classifier_balanced_accuracy']:8.3f} "
              f"{(f'{auc:.3f}' if auc is not None else 'n/a'):>6s} "
              f"{m['margin_mae_db']:7.2f} {m['margin_rmse_db']:8.2f} "
              f"{m['margin_r2']:7.4f}")

    ablation = report.get("design_only_ablation") or {}
    if ablation:
        print("\ndesign-only ablation (6 design parameters, no spectral features):")
        for m in ablation["band_metrics"]:
            auc = m["classifier_roc_auc"]
            print(f"{m['band_label']:18s} {m['fail_rate']:6.1%} "
                  f"{m['classifier_accuracy']:6.3f} "
                  f"{m['classifier_balanced_accuracy']:8.3f} "
                  f"{(f'{auc:.3f}' if auc is not None else 'n/a'):>6s} "
                  f"{m['margin_mae_db']:7.2f} {m['margin_rmse_db']:8.2f} "
                  f"{m['margin_r2']:7.4f}")

    audit = report["monotonicity_audit"]  # type: ignore[index]
    violations = [a for a in audit if not a["respects_constraint"]]
    print(f"\nmonotonicity     : {len(audit) - len(violations)}/{len(audit)} "
          f"design features respect their physics constraint")
    for v in violations:
        print(f"  VIOLATION {v['feature']} ({v['worst_risk_decrease']:+.3e})")

    stability = report.get("seed_stability") or {}
    if stability:
        print(f"seed stability   : simulator {stability['simulator_margin_sd_db']:.2f} dB SD "
              f"vs surrogate {stability['model_margin_sd_db']:.2f} dB SD "
              f"({stability['variance_reduction_factor']:.1f}x tighter)")
    print("=" * 74)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--skip-stability", action="store_true",
                        help="skip the (slow) simulator-vs-surrogate stability audit")
    parser.add_argument("--report-only", action="store_true",
                        help="print the saved validation report and exit")
    args = parser.parse_args()

    if args.report_only:
        if not REPORT_PATH.exists():
            raise SystemExit(f"no report at {REPORT_PATH}; train first")
        print_report(json.loads(REPORT_PATH.read_text(encoding="utf-8")))
        return

    report = train(args.n_samples, args.jobs, skip_stability=args.skip_stability)
    print_report(report)


if __name__ == "__main__":
    main()
