"""Unit tests for the closed-form trade-off proxies and Pareto flags.

The full 27-point EMC sweep is covered by a skipped-if-no-model API test with
a short grid; these tests do not call the simulator.
"""

from tradeoff import (
    F_SW_MAX_KHZ,
    F_SW_MIN_KHZ,
    acoustic_risk,
    pareto_mask,
    ripple_cost,
    ripple_peak_to_peak_a,
)


def test_ripple_cost_falls_as_frequency_rises():
    """Lower carrier → more current ripple. The EMC trade-off."""
    low = ripple_cost(F_SW_MIN_KHZ)
    mid = ripple_cost(8.0)
    high = ripple_cost(F_SW_MAX_KHZ)
    assert 0.0 <= high < mid < low <= 100.0


def test_ripple_peak_scales_inversely_with_frequency():
    a = ripple_peak_to_peak_a(4.0)
    b = ripple_peak_to_peak_a(8.0)
    assert abs(a / b - 2.0) < 1e-9


def test_acoustic_risk_falls_as_frequency_rises():
    low = acoustic_risk(F_SW_MIN_KHZ)
    mid = acoustic_risk(8.0)
    high = acoustic_risk(F_SW_MAX_KHZ)
    assert 0.0 <= high < mid < low <= 100.0
    assert abs(mid - 50.0) < 0.05


def test_ripple_and_acoustic_move_together_against_frequency():
    """Both named costs of dropping the carrier rise as f_sw falls."""
    assert ripple_cost(4.0) > ripple_cost(12.0)
    assert acoustic_risk(4.0) > acoustic_risk(12.0)


def test_pareto_drops_dominated_aligned_objectives():
    # Score falls, cost rises: every point is worse on both axes as index grows,
    # so only the first point (best score AND best cost) is Pareto.
    scores = [90.0, 70.0, 50.0]
    costs = [10.0, 40.0, 80.0]
    assert pareto_mask(scores, costs) == [True, False, False]


def test_pareto_keeps_true_tradeoff():
    # Score falls, cost also falls: each point trades headroom for a cheaper
    # second objective, so the whole chain is Pareto.
    scores = [90.0, 70.0, 50.0]
    costs = [80.0, 40.0, 10.0]
    assert pareto_mask(scores, costs) == [True, True, True]


def test_render_tradeoff_png_smoke():
    from certificate import render_tradeoff_png

    points = [
        {
            "switching_frequency_khz": 4.0,
            "emc_risk_score": 80.0,
            "acoustic_risk": 83.0,
            "pareto_acoustic": True,
        },
        {
            "switching_frequency_khz": 8.0,
            "emc_risk_score": 65.0,
            "acoustic_risk": 50.0,
            "pareto_acoustic": True,
        },
        {
            "switching_frequency_khz": 7.0,
            "emc_risk_score": 64.0,
            "acoustic_risk": 60.0,
            "pareto_acoustic": False,
        },
    ]
    png = render_tradeoff_png(points, 8.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
