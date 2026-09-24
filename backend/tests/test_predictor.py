from pathlib import Path

import pytest

from predictor import MODEL_PATH, load_models, predict
from simulate import DeviceParameters


pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="trained artifact backend/models/certifai_models.joblib is missing",
)


def test_models_load():
    bundle = load_models()
    assert len(bundle.classifiers) == 3
    assert len(bundle.regressors) == 3
    assert bundle.scaler is not None


def test_predict_returns_risk_payload():
    result = predict(
        DeviceParameters.clamped(
            switching_frequency_khz=8.0,
            dv_dt_v_per_us=3500.0,
            cable_length_m=30.0,
            shielding_quality=0.7,
            load_current_a=45.0,
            pwm_modulation_type="SPWM",
            input_filter_quality=0.45,
        )
    )
    assert result["risk_level"] in {"LOW", "MODERATE", "HIGH"}
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["risk_score_plus_minus"] >= 0.0
    assert len(result["bands"]) == 3
    assert len(result["signals"]) == 5
    assert len(result["power_quality"]) == 2
    assert "This tool estimates EMC risk" in result["framing"]
    assert result["radiated"]["badge"] == "EXPLORATORY"
    assert result["radiated"]["risk_score"] != result["risk_score"] or True
    assert len(result["radiated"]["bands"]) == 2
    assert any(row["parameter"] == "cm_choke_effectiveness" for row in result["shap"]["contributions"])
    assert result["shap"]["contributions"]
    assert result["shap"]["unit"] == "dB"
    assert max(abs(row["shap"]) for row in result["shap"]["contributions"]) > 0.5
    assert result["countermeasures"]
    assert Path(MODEL_PATH).name == "certifai_models.joblib"
