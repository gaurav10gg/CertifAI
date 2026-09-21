from simulate import (
    EXPLORER_POINTS,
    PQ_N_SAMPLES,
    PQ_SAMPLE_RATE_HZ,
    DeviceParameters,
    simulate_device,
)
import numpy as np


def _params(**overrides):
    values = dict(
        switching_frequency_khz=8.0,
        dv_dt_v_per_us=3500.0,
        cable_length_m=30.0,
        shielding_quality=0.7,
        load_current_a=45.0,
        pwm_modulation_type="SPWM",
        input_filter_quality=0.45,
    )
    values.update(overrides)
    return DeviceParameters.clamped(**values)


def _switching_ripple(current: np.ndarray, sample_rate_hz: float) -> float:
    """RMS after removing DC and the 50 Hz fundamental."""
    n = current.size
    spectrum = np.fft.rfft(current)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    spectrum[np.abs(freqs) < 80.0] = 0.0
    residual = np.fft.irfft(spectrum, n=n)
    return float(residual.std())


def test_simulate_five_explorer_traces():
    sim = simulate_device(_params(), seed=1)
    keys = [trace.key for trace in sim.explorer]
    assert keys == [
        "dc_link",
        "motor_voltage",
        "motor_current",
        "common_mode",
        "input_current",
    ]
    for trace in sim.explorer:
        assert 8 <= trace.time_s.size <= EXPLORER_POINTS + 2
        assert trace.values.size == trace.time_s.size


def test_simulate_power_quality_shapes():
    sim = simulate_device(_params(), seed=2)
    assert sim.pq_motor_current_a.shape == (PQ_N_SAMPLES,)
    assert sim.pq_input_current_a.shape == (PQ_N_SAMPLES,)
    assert sim.pq_dc_link_v.shape == (PQ_N_SAMPLES,)
    assert sim.measured_segments.ndim == 2
    assert sim.measured_segments.shape[0] >= 1


def test_motor_current_smoother_at_higher_fsw():
    slow = simulate_device(_params(switching_frequency_khz=4.0), seed=3)
    fast = simulate_device(_params(switching_frequency_khz=16.0), seed=3)
    ripple_slow = _switching_ripple(slow.pq_motor_current_a, PQ_SAMPLE_RATE_HZ)
    ripple_fast = _switching_ripple(fast.pq_motor_current_a, PQ_SAMPLE_RATE_HZ)
    assert ripple_fast < ripple_slow
