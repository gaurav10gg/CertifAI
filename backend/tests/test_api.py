from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from predictor import MODEL_PATH


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "models_loaded" in body


def test_devices_and_methodology_smoke():
    devices = client.get("/api/devices")
    assert devices.status_code == 200
    body = devices.json()
    assert body["devices"]
    fsw = next(p for p in body["parameters"] if p["key"] == "switching_frequency_khz")
    assert fsw["min"] == 3.0
    assert fsw["max"] == 16.0
    assert "framing" in body or "disclaimer" in body

    methodology = client.get("/api/methodology")
    assert methodology.status_code == 200
    method = methodology.json()
    assert "does not predict" in method["framing"].lower() or "risk" in method["disclaimer"].lower()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model artifact missing")
def test_validation_endpoint_serves_measured_headlines():
    from validation import SUITE_PATH

    if not SUITE_PATH.exists():
        pytest.skip("validation_suite.json has not been generated yet")
    response = client.get("/api/validation")
    assert response.status_code == 200
    body = response.json()
    assert body["passed"] is True
    assert "200 randomized configurations" in body["headlines"]["monotonicity"]
    assert "mean error" in body["headlines"]["shap_additivity"]
    assert body["monotonicity"]["n_configs"] == 200
    assert body["shap_additivity"]["mean_abs_error_db"] < 1e-4


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model artifact missing")
def test_predict_smoke():
    response = client.post(
        "/api/predict",
        json={
            "parameters": {
                "switching_frequency_khz": 8,
                "dv_dt_v_per_us": 3500,
                "cable_length_m": 30,
                "shielding_quality": 0.7,
                "load_current_a": 45,
                "pwm_modulation_type": "SPWM",
                "input_filter_quality": 0.45,
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] in {"LOW", "MODERATE", "HIGH"}
    assert Path(MODEL_PATH).exists()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model artifact missing")
def test_tradeoff_sweep_smoke():
    response = client.post(
        "/api/tradeoff",
        json={
            "parameters": {
                "switching_frequency_khz": 8,
                "dv_dt_v_per_us": 3000,
                "cable_length_m": 30,
                "shielding_quality": 0.78,
                "load_current_a": 45,
                "pwm_modulation_type": "SPWM",
                "input_filter_quality": 0.45,
            },
            "n_points": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 5
    assert body["points"][0]["switching_frequency_khz"] == 3.0
    assert body["points"][-1]["switching_frequency_khz"] == 16.0
    required = {
        "switching_frequency_khz",
        "emc_risk_score",
        "ripple_cost",
        "acoustic_risk",
        "pareto_ripple",
        "pareto_acoustic",
    }
    assert required.issubset(body["points"][0].keys())
    assert "efficiency_cost" not in body["points"][0]
    assert "switching_loss_proxy_w" not in body["points"][0]
    assert sum(1 for p in body["points"] if p["is_current"]) == 1
    freqs = [p["switching_frequency_khz"] for p in body["points"]]
    assert freqs == sorted(freqs)
    acoustics = [p["acoustic_risk"] for p in body["points"]]
    assert acoustics == sorted(acoustics, reverse=True)
    ripples = [p["ripple_cost"] for p in body["points"]]
    assert ripples == sorted(ripples, reverse=True)
