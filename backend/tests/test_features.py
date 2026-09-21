import numpy as np

from features import _dominant_statement, _harmonic_peaks, extract_features
from simulate import PQ_N_SAMPLES, PQ_SAMPLE_RATE_HZ, DeviceParameters, simulate_device


def test_thd_on_known_fifth_harmonic():
    t = np.arange(PQ_N_SAMPLES) / PQ_SAMPLE_RATE_HZ
    waveform = np.sin(2 * np.pi * 50.0 * t) + 0.30 * np.sin(2 * np.pi * 250.0 * t)
    thd, fund, peaks = _harmonic_peaks(waveform, PQ_SAMPLE_RATE_HZ, 50.0)
    assert 24.0 < thd < 33.0
    assert fund > 0.5
    assert peaks[0].order == 5
    orders, statement = _dominant_statement("input current", peaks)
    assert 5 in orders
    assert "5th" in statement


def test_extract_features_keeps_emc_and_pq_separate():
    params = DeviceParameters.clamped(
        switching_frequency_khz=8.0,
        dv_dt_v_per_us=3500.0,
        cable_length_m=30.0,
        shielding_quality=0.7,
        load_current_a=45.0,
        pwm_modulation_type="SPWM",
        input_filter_quality=0.8,
    )
    bundle = extract_features(simulate_device(params, seed=4))
    assert bundle.feature_vector.shape == (18,)
    assert len(bundle.bands) == 3
    keys = {report.key for report in bundle.power_quality}
    assert keys == {"motor_current", "input_current"}
    for report in bundle.power_quality:
        assert report.thd_percent >= 0.0
        assert len(report.peaks) <= 5
