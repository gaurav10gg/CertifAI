"""Train and validate the separate radiated-emissions models.

Run from backend/:

    .venv\\Scripts\\python.exe tools\\train_radiated.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import xgboost_monotone_string
from radiated import RADIATED_DESIGN_FEATURES, radiated_features, radiated_margins
from simulate import DeviceParameters, sample_random_parameters

MODEL_PATH = ROOT / "models" / "radiated_models.joblib"
N_SAMPLES = 4000
SEED = 20260924


def _score(margins: np.ndarray) -> float:
    from predictor import _compliance_score
    return float(_compliance_score(margins))


def train() -> dict:
    rng = np.random.default_rng(SEED)
    designs = [sample_random_parameters(rng) for _ in range(N_SAMPLES)]
    x = np.vstack([radiated_features(p)[0] for p in designs])
    margins = np.vstack([radiated_margins(p) for p in designs])
    y_fail = (margins <= 0.0).astype(float)

    idx = np.arange(N_SAMPLES)
    idx_train, idx_temp = train_test_split(idx, test_size=0.3, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=SEED)
    scaler = StandardScaler().fit(x[idx_train])
    signs = tuple(spec[1] for spec in RADIATED_DESIGN_FEATURES)
    margin_signs = tuple(-s for s in signs)

    regressors = []
    maes = []
    for band in range(margins.shape[1]):
        reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            monotone_constraints=xgboost_monotone_string(margin_signs),
            tree_method="hist",
            random_state=SEED,
        )
        reg.fit(scaler.transform(x[idx_train]), margins[idx_train, band])
        pred = reg.predict(scaler.transform(x[idx_test]))
        maes.append(float(mean_absolute_error(margins[idx_test, band], pred)))
        regressors.append(reg)

    bundle = {
        "scaler": scaler,
        "regressors": regressors,
        "feature_names": [spec[0] for spec in RADIATED_DESIGN_FEATURES],
        "margin_signs": margin_signs,
        "test_mae_db": maes,
        "n_samples": N_SAMPLES,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    return bundle


def validate(bundle: dict) -> None:
    scaler = bundle["scaler"]
    regressors = bundle["regressors"]
    names = bundle["feature_names"]
    signs = bundle["margin_signs"]
    rng = np.random.default_rng(SEED + 1)
    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    n_configs, n_steps = 200, 8
    violations = 0
    checks = 0
    for _ in range(n_configs):
        base = mean + rng.normal(0.0, 1.0, size=mean.shape) * scale
        for index, sign in enumerate(signs):
            if sign == 0:
                continue
            grid = np.linspace(mean[index] - 2 * scale[index], mean[index] + 2 * scale[index], n_steps)
            rows = np.repeat(base.reshape(1, -1), n_steps, axis=0)
            rows[:, index] = grid
            scaled = scaler.transform(rows)
            margins = np.column_stack([reg.predict(scaled) for reg in regressors])
            scores = np.array([_score(row) for row in margins])
            # Margin sign -1 means the feature raises risk, so the score must fall.
            delta = np.diff(scores)
            if sign < 0 and np.any(delta > 1e-6):
                violations += int(np.sum(delta > 1e-6))
            if sign > 0 and np.any(delta < -1e-6):
                violations += int(np.sum(delta < -1e-6))
            checks += n_steps - 1

    shap_errors = []
    for _ in range(200):
        row = mean + rng.normal(0.0, 1.0, size=mean.shape) * scale
        scaled = scaler.transform(row.reshape(1, -1))
        for reg in regressors:
            contrib = reg.get_booster().predict(xgb.DMatrix(scaled), pred_contribs=True)[0]
            shap_errors.append(abs(float(contrib[:-1].sum() + contrib[-1] - reg.predict(scaled)[0])))

    print(f"radiated test MAE dB: {bundle['test_mae_db']}")
    print(f"radiated monotonicity violations: {violations} / {checks}")
    print(f"radiated SHAP mean abs error dB: {float(np.mean(shap_errors)):.3e}")
    print("feature order:", names)


if __name__ == "__main__":
    validate(train())
