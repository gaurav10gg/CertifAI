"""
calibrate.py -- residual correction from a small measured dataset.

Real EMC pre-compliance data arrives in tens of points, not thousands, so it
must not be dumped into ``train_model.py`` where it would be numerically drowned.
This module keeps the physics-derived models as the prior and learns a small,
clamped, monotonically constrained correction on top of them.

The implementation is complete and unit-tested against synthetic CSVs. It has
never been run against chamber data -- until it has, every figure the tool
reports is still relative to the simulator, and the UI says so.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from features import DESIGN_FEATURE_SPECS, FEATURE_NAMES, EMC_BANDS, extract_features
from predictor import MODEL_PATH, load_models
from simulate import DeviceParameters, simulate_device

REQUIRED_COLUMNS: Sequence[str] = (
    "switching_frequency_khz",
    "dv_dt_v_per_us",
    "cable_length_m",
    "shielding_quality",
    "load_current_a",
    "pwm_modulation_type",
    "measured_peak_dbuv_band_a",
    "measured_peak_dbuv_band_b",
    "measured_peak_dbuv_band_c",
    "limit_dbuv_band_a",
    "limit_dbuv_band_b",
    "limit_dbuv_band_c",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in REQUIRED_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        return list(reader)


def _row_params(row: Dict[str, str]) -> DeviceParameters:
    return DeviceParameters.clamped(
        switching_frequency_khz=float(row["switching_frequency_khz"]),
        dv_dt_v_per_us=float(row["dv_dt_v_per_us"]),
        cable_length_m=float(row["cable_length_m"]),
        shielding_quality=float(row["shielding_quality"]),
        load_current_a=float(row["load_current_a"]),
        pwm_modulation_type=row["pwm_modulation_type"],
        input_filter_quality=float(row.get("input_filter_quality") or 0.45),
    )


def _design_matrix(params_list: Sequence[DeviceParameters]) -> np.ndarray:
    columns = []
    for spec in DESIGN_FEATURE_SPECS:
        if spec.name == "pwm_cm_penalty_db":
            columns.append([p.pwm_cm_penalty_db for p in params_list])
        else:
            columns.append([getattr(p, spec.name) for p in params_list])
    return np.column_stack(columns)


def calibrate_from_measurements(
    measurements_csv: Path,
    *,
    model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    max_correction_db: float = 12.0,
    ridge_alpha: float = 8.0,
) -> Dict[str, Any]:
    """Fit a per-band residual model and write a calibrated artifact bundle.

    Steps
    -----
    1. Load the CSV and rebuild each row's ``DeviceParameters``.
    2. Re-run the physics chain to obtain the simulator's own peak per band.
    3. Residual = measured_margin - simulated_margin, where
       margin = limit - peak.
    4. Ridge regression on the six design parameters, coefficients clipped so
       they cannot invert a known monotonic relationship.
    5. Clamp every predicted residual to +/- ``max_correction_db``.
    6. Leave-one-out RMSE of the corrected margin is written into
       ``margin_rmse_db`` so confidence figures reflect hardware error once
       real data exists.
    """
    rows = _read_csv(Path(measurements_csv))
    if len(rows) < 4:
        raise ValueError("Need at least 4 measured configurations to fit a residual.")

    models = load_models(str(model_path) if model_path else None)
    params_list = [_row_params(row) for row in rows]
    simulated_margins = np.zeros((len(rows), len(EMC_BANDS)))
    measured_margins = np.zeros_like(simulated_margins)

    for i, (row, params) in enumerate(zip(rows, params_list)):
        bundle = extract_features(simulate_device(params))
        for j, band in enumerate(EMC_BANDS):
            suffix = band.key[-1]  # a/b/c
            measured_peak = float(row[f"measured_peak_dbuv_band_{suffix}"])
            limit = float(row[f"limit_dbuv_band_{suffix}"])
            simulated_margins[i, j] = bundle.bands[j].margin_db
            measured_margins[i, j] = limit - measured_peak

    residual = measured_margins - simulated_margins
    residual = np.clip(residual, -max_correction_db, max_correction_db)
    X = _design_matrix(params_list)

    corrections: List[Dict[str, Any]] = []
    loo_rmse: List[float] = []
    for j, band in enumerate(EMC_BANDS):
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X, residual[:, j])
        # Preserve physics: a coefficient may not push against the known sign.
        coef = ridge.coef_.copy()
        for k, spec in enumerate(DESIGN_FEATURE_SPECS):
            # Residual is in margin (higher = safer). A feature that raises
            # risk (risk_sign +1) must not receive a positive margin coefficient
            # large enough to invert the prior -- we only allow coefficients
            # whose sign agrees with -risk_sign, or we shrink them to zero.
            allowed = -spec.risk_sign
            if coef[k] * allowed < 0:
                coef[k] = 0.0
        ridge.coef_ = coef

        # Leave-one-out RMSE of the *corrected* margin.
        sqerr = []
        for hold in range(len(rows)):
            mask = np.ones(len(rows), dtype=bool)
            mask[hold] = False
            fold = Ridge(alpha=ridge_alpha)
            fold.fit(X[mask], residual[mask, j])
            pred_res = float(np.clip(fold.predict(X[hold : hold + 1])[0],
                                     -max_correction_db, max_correction_db))
            corrected = simulated_margins[hold, j] + pred_res
            sqerr.append((corrected - measured_margins[hold, j]) ** 2)
        rmse = float(np.sqrt(np.mean(sqerr)))
        loo_rmse.append(rmse)
        corrections.append({
            "band": band.key,
            "intercept": float(ridge.intercept_),
            "coefficients": {
                spec.name: float(coef[k]) for k, spec in enumerate(DESIGN_FEATURE_SPECS)
            },
            "residual_std_db": float(np.std(residual[:, j])),
            "loo_rmse_db": rmse,
        })

    artifact = joblib.load(model_path or MODEL_PATH)
    artifact["calibration"] = {
        "source_csv": str(measurements_csv),
        "n_measurements": len(rows),
        "max_correction_db": max_correction_db,
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bands": corrections,
        "note": (
            "Residual correction on top of the synthetic prior. Absolute "
            "levels after calibration are still only as good as the measured "
            "set, which is small by construction."
        ),
    }
    artifact["margin_rmse_db"] = loo_rmse
    report = artifact.get("report") or {}
    report["calibration"] = artifact["calibration"]
    artifact["report"] = report

    destination = Path(output_path) if output_path else (
        (model_path or MODEL_PATH).with_name("certifai_models_calibrated.joblib")
    )
    joblib.dump(artifact, destination, compress=3)

    return {
        "output_path": str(destination),
        "n_measurements": len(rows),
        "loo_rmse_db": loo_rmse,
        "bands": corrections,
        "feature_names": list(FEATURE_NAMES[:6]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Measured configurations CSV")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = calibrate_from_measurements(args.csv, output_path=args.output)
    print(summary)
