from predictor import MODEL_PATH, load_models
from validation import check_monotonicity, check_shap_additivity, load_suite

import pytest

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="trained artifact backend/models/certifai_models.joblib is missing",
)


def test_monotonicity_holds_on_a_small_probe():
    bundle = load_models()
    report = check_monotonicity(bundle, n_configs=8, n_steps=5)
    assert report["n_checks"] == 8 * 6
    assert report["passed"], report.get("first_failures")
    assert report["consistent_percent"] == 100.0


def test_shap_adds_up_to_the_prediction():
    bundle = load_models()
    report = check_shap_additivity(bundle, n_designs=12)
    assert report["n_checks"] == 12 * 3
    assert report["passed"]
    assert report["mean_abs_error_db"] < 1e-4


def test_committed_suite_json_is_a_pass():
    suite = load_suite()
    if suite is None:
        pytest.skip("validation_suite.json has not been generated yet")
    assert suite["passed"]
    assert suite["monotonicity"]["n_configs"] == 200
    assert suite["monotonicity"]["n_violations"] == 0
    assert suite["shap_additivity"]["mean_abs_error_db"] < 1e-4
    assert "200 randomized configurations" in suite["headlines"]["monotonicity"]
    assert "mean error" in suite["headlines"]["shap_additivity"]
