"""
features.py -- FFT / spectral feature extraction and limit-curve comparison.

This module turns a time-domain :class:`~simulate.SimulationResult` into

  * a receiver-emulated emission trace (dBuV vs frequency) for charting,
  * four descriptive features per standard conducted-emission band,
  * the margin-to-limit per band, which is the label used by ``train_model.py``,
  * the ordered feature vector plus the monotone-constraint vector consumed by
    the physics-informed XGBoost models.

===============================================================================
LIMIT CURVE -- READ THIS BEFORE TRUSTING ANY NUMBER THIS TOOL PRODUCES
===============================================================================
``LIMIT_CURVE_ANCHORS`` below is a **SYNTHETIC, EN-12016-STYLE** limit line. It
is *not* the normative EN 12016 limit table.

Why: EN 12016 ("Electromagnetic compatibility -- Product family standard for
lifts, escalators and moving walks -- Immunity") and the emission standards it
references are copyrighted documents whose normative tables are not publicly
redistributable. We therefore construct a limit line that is *shaped like* the
publicly-described conducted-emission limits that lift-sector equipment is
generally assessed against (CISPR 11 / EN 55011 Group 1 Class A quasi-peak
levels of 79 dBuV in the 150-500 kHz region and 73 dBuV above it, which
EN 61000-6-4 broadly follows), and we add our own smooth tapers so the curve is
continuous rather than stepped.

Consequences that must be surfaced to the user:
  * Risk scores from this tool are relative to OUR curve.
  * A LOW RISK result is not a statement about EN 12016 conformity.
  * The tapers at 150-500 kHz and 5-30 MHz are our additions, not standard.

This assumption is mirrored verbatim on the application's Methodology page.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Final, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from simulate import (
    ML_PARAMETER_KEYS,
    PARAMETER_RANGE_BY_KEY,
    SimulationResult,
)

# ---------------------------------------------------------------------------
# Conducted-emission bands (CISPR band B is 150 kHz - 30 MHz; we subdivide it
# into the three sub-ranges the drive industry usually reports against).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmcBand:
    key: str
    label: str
    f_low_hz: float
    f_high_hz: float


EMC_BANDS: Final[Tuple[EmcBand, ...]] = (
    EmcBand("band_a", "150 kHz - 500 kHz", 150e3, 500e3),
    EmcBand("band_b", "500 kHz - 5 MHz", 500e3, 5e6),
    EmcBand("band_c", "5 MHz - 30 MHz", 5e6, 30e6),
)

ANALYSIS_F_LOW_HZ: Final[float] = EMC_BANDS[0].f_low_hz
ANALYSIS_F_HIGH_HZ: Final[float] = EMC_BANDS[-1].f_high_hz

# CISPR 16-1-1 resolution bandwidth for band B.
RECEIVER_RBW_HZ: Final[float] = 9e3

# SYNTHETIC limit line -- see the module docstring. (frequency Hz, level dBuV),
# log-linearly interpolated between anchors.
LIMIT_CURVE_ANCHORS: Final[Tuple[Tuple[float, float], ...]] = (
    (150e3, 79.0),   # anchored on CISPR 11 Group 1 Class A quasi-peak
    (500e3, 73.0),   # taper is our own addition (standard curve is stepped)
    (5e6, 73.0),
    (30e6, 70.0),    # our own tightening, to reflect shaft radiated coupling
)

LIMIT_CURVE_DESCRIPTION: Final[str] = (
    "Synthetic EN-12016-style conducted-emission limit: 79 dBuV at 150 kHz "
    "tapering log-linearly to 73 dBuV at 500 kHz, flat at 73 dBuV to 5 MHz, "
    "then tapering to 70 dBuV at 30 MHz. Anchored on publicly described "
    "CISPR 11 Group 1 Class A quasi-peak levels; the tapers are an assumption "
    "of this tool and are not normative."
)

# A spectral line counts as a "harmonic of concern" once it comes within this
# many dB of the limit line.
HARMONIC_PROXIMITY_DB: Final[float] = 10.0

# Number of log-spaced points returned for the frontend chart.
CHART_POINTS: Final[int] = 220


# ---------------------------------------------------------------------------
# Feature schema + physics-informed monotone constraints
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """One model input, together with its known physical sign.

    ``risk_sign`` is the sign of the feature's effect on **failure risk**:
        +1  increasing this feature can only increase the risk of exceeding the limit
        -1  increasing this feature can only decrease that risk

    ``train_model.py`` feeds these signs straight into XGBoost's
    ``monotone_constraints``. For the *margin* regressor (where a higher target
    means more headroom, i.e. less risk) every sign is negated.
    """

    name: str
    risk_sign: int
    rationale: str


DESIGN_FEATURE_SPECS: Final[Tuple[FeatureSpec, ...]] = (
    FeatureSpec(
        "switching_frequency_khz", +1,
        "For an impulsive pulse train the amplitude of each spectral line scales "
        "with the pulse repetition frequency, so raising the carrier lifts the "
        "whole 150 kHz-30 MHz envelope.",
    ),
    FeatureSpec(
        "dv_dt_v_per_us", +1,
        "Faster edges move the trapezoid spectrum's second corner frequency "
        "upward, adding high-frequency energy without removing any.",
    ),
    FeatureSpec(
        "cable_length_m", +1,
        "Parasitic capacitance to earth grows with length, so the common-mode "
        "displacement current i = C dv/dt grows with it.",
    ),
    FeatureSpec(
        "shielding_quality", -1,
        "Screen insertion loss and common-mode choking are strictly "
        "attenuating; better shielding can never raise the emission.",
    ),
    FeatureSpec(
        "load_current_a", +1,
        "Higher motor current raises circulating and return-path currents, "
        "lifting emission amplitude.",
    ),
    FeatureSpec(
        "pwm_cm_penalty_db", +1,
        "Modelled common-mode penalty of the modulation strategy in dB, relative "
        "to sinusoidal PWM. Driven mainly by commutation count (discontinuous "
        "PWM removes about a third of all commutations) and by whether the "
        "strategy injects a zero-sequence component.",
    ),
)

_BAND_FEATURE_TEMPLATES: Final[Tuple[Tuple[str, int, str], ...]] = (
    ("peak_dbuv", +1,
     "Highest receiver-emulated level in the band; directly compared against "
     "the limit line."),
    ("rms_dbuv", +1,
     "Total energy in the band; more energy cannot reduce the exceedance risk."),
    ("harmonic_count", +1,
     "Number of resolved lines within "
     f"{HARMONIC_PROXIMITY_DB:.0f} dB of the limit; more near-limit lines means "
     "more ways to fail."),
    ("thd_score", +1,
     "THD-like measure of how far switching-related content sits above the "
     "ambient noise floor in the band."),
)

BAND_FEATURE_SPECS: Final[Tuple[FeatureSpec, ...]] = tuple(
    FeatureSpec(f"{band.key}_{suffix}", sign, rationale)
    for band in EMC_BANDS
    for suffix, sign, rationale in _BAND_FEATURE_TEMPLATES
)

FEATURE_SPECS: Final[Tuple[FeatureSpec, ...]] = DESIGN_FEATURE_SPECS + BAND_FEATURE_SPECS
FEATURE_NAMES: Final[Tuple[str, ...]] = tuple(s.name for s in FEATURE_SPECS)

# Physics-informed monotone constraint vector, in feature order.
#   * RISK targets  (P(fail) classifier)  use these signs as-is.
#   * MARGIN target (headroom regressor)  uses the negated vector.
# Deliberately NOT hidden in a config file -- this vector *is* the
# physics-informed part of the model and is surfaced in the UI.
MONOTONE_CONSTRAINTS_RISK: Final[Tuple[int, ...]] = tuple(
    s.risk_sign for s in FEATURE_SPECS
)
MONOTONE_CONSTRAINTS_MARGIN: Final[Tuple[int, ...]] = tuple(
    -s.risk_sign for s in FEATURE_SPECS
)


def xgboost_monotone_string(constraints: Sequence[int]) -> str:
    """Render a constraint vector in XGBoost's ``(1,0,-1,...)`` notation."""
    return "(" + ",".join(str(int(c)) for c in constraints) + ")"


# ---------------------------------------------------------------------------
# Limit curve
# ---------------------------------------------------------------------------
def limit_dbuv(freq_hz: np.ndarray | float) -> np.ndarray:
    """Synthetic EN-12016-style limit level at the given frequencies.

    Log-linear interpolation between :data:`LIMIT_CURVE_ANCHORS`, held flat
    outside the 150 kHz-30 MHz analysis range.
    """
    freqs = np.atleast_1d(np.asarray(freq_hz, dtype=float))
    anchor_f = np.log10([a[0] for a in LIMIT_CURVE_ANCHORS])
    anchor_l = np.array([a[1] for a in LIMIT_CURVE_ANCHORS])
    safe = np.clip(freqs, 1.0, None)
    return np.interp(np.log10(safe), anchor_f, anchor_l)


# ---------------------------------------------------------------------------
# Results containers
# ---------------------------------------------------------------------------
@dataclass
class BandAnalysis:
    """Per-band spectral summary and margin against the synthetic limit."""

    key: str
    label: str
    f_low_hz: float
    f_high_hz: float
    peak_dbuv: float
    peak_frequency_hz: float
    rms_dbuv: float
    harmonic_count: int
    thd_score: float
    limit_at_peak_dbuv: float
    margin_db: float          # limit - emission, minimised over the band; >0 = pass
    worst_frequency_hz: float
    passes: bool


@dataclass
class SpectrumTrace:
    """Down-sampled, log-spaced trace for plotting."""

    frequency_hz: np.ndarray
    emission_dbuv: np.ndarray
    limit_dbuv: np.ndarray


@dataclass
class HarmonicPeak:
    order: int
    frequency_hz: float
    amplitude: float
    percent_of_fundamental: float


@dataclass
class PowerQualityReport:
    """THD / harmonic summary for one current waveform.

    Separate from the conducted-emission band analysis: this answers a power-
    quality question (how distorted is the current?), not an EMC-limit question.
    """

    key: str
    label: str
    thd_percent: float
    fundamental_hz: float
    fundamental_amplitude: float
    peaks: List[HarmonicPeak]
    dominant_orders: Tuple[int, ...]
    dominant_statement: str


@dataclass
class FeatureBundle:
    bands: List[BandAnalysis]
    spectrum: SpectrumTrace
    feature_vector: np.ndarray
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    diagnostics: Dict[str, float] = field(default_factory=dict)
    power_quality: List[PowerQualityReport] = field(default_factory=list)

    def band_by_key(self, key: str) -> BandAnalysis:
        for band in self.bands:
            if band.key == key:
                return band
        raise KeyError(key)

    def as_feature_dict(self) -> Dict[str, float]:
        return dict(zip(self.feature_names, self.feature_vector.tolist()))


# ---------------------------------------------------------------------------
# Spectrum computation
# ---------------------------------------------------------------------------
def _amplitude_spectrum(
    segments: np.ndarray, sample_rate_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Max-held single-sided amplitude spectrum in volts.

    Each row of ``segments`` is one receiver dwell window at a different phase of
    the motor fundamental (see the segmentation note in ``simulate.py``). Taking
    the per-frequency maximum across windows reproduces the max-hold behaviour of
    a swept EMI receiver, and removes the several-dB capture-phase variance that
    a single window would otherwise show.

    Amplitudes are normalised as ``2 |X_k| / N`` so a pure tone of amplitude A
    produces a line of height A. This normalisation is what makes spectral line
    height scale with the pulse repetition frequency for an impulsive train --
    the standard EMC result that emissions rise with switching frequency.
    """
    segments = np.atleast_2d(segments)
    n = segments.shape[1]
    window = np.hanning(n)
    # Coherent gain compensation so windowed line heights stay calibrated.
    coherent_gain = window.mean()
    spectra = np.fft.rfft(segments * window[None, :], axis=1)
    amplitude = (2.0 / (n * coherent_gain)) * np.abs(spectra)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    return freqs, amplitude.max(axis=0)


def _receiver_trace(
    freqs: np.ndarray, amplitude_v: np.ndarray, sample_rate_hz: float, n_samples: int
) -> np.ndarray:
    """Emulate a swept EMI receiver: integrate power over the 9 kHz RBW.

    Summing power across the resolution bandwidth is what a real receiver's IF
    filter does. A narrowband line keeps its level; a broadband noise floor
    rises by 10*log10(bins per RBW), exactly as measured floors do.

    Quasi-peak weighting note: the CISPR quasi-peak detector reads below the
    peak detector only for pulse repetition rates well below ~100 Hz. Inverter
    switching produces repetition rates of tens of kHz, so the detector fully
    charges and QP ~= peak. The correction is therefore applied but is
    negligible here; it is retained so the model stays correct if low-PRF
    sources are added later.
    """
    bin_hz = sample_rate_hz / n_samples
    rbw_bins = max(1, int(round(RECEIVER_RBW_HZ / bin_hz)))
    power = amplitude_v ** 2
    integrated = uniform_filter1d(power, size=rbw_bins, mode="nearest") * rbw_bins
    return 10.0 * np.log10(np.maximum(integrated, 1e-30)) + 120.0  # -> dBuV


def _detect_lines(band_amp_dbuv: np.ndarray, threshold_dbuv: np.ndarray) -> np.ndarray:
    """Indices of local maxima that rise above a per-bin threshold."""
    if band_amp_dbuv.size < 3:
        return np.empty(0, dtype=int)
    interior = band_amp_dbuv[1:-1]
    is_peak = (interior >= band_amp_dbuv[:-2]) & (interior > band_amp_dbuv[2:])
    above = interior > threshold_dbuv[1:-1]
    return np.flatnonzero(is_peak & above) + 1


def _analyse_band(
    band: EmcBand,
    freqs: np.ndarray,
    trace_dbuv: np.ndarray,
    amplitude_v: np.ndarray,
) -> BandAnalysis:
    mask = (freqs >= band.f_low_hz) & (freqs < band.f_high_hz)
    band_freqs = freqs[mask]
    band_trace = trace_dbuv[mask]
    band_amp = amplitude_v[mask]
    band_limit = limit_dbuv(band_freqs)

    peak_idx = int(np.argmax(band_trace))
    peak_dbuv = float(band_trace[peak_idx])

    # Band RMS of the calibrated amplitude spectrum, expressed in dBuV.
    rms_v = float(np.sqrt(np.mean(band_amp ** 2)))
    rms_dbuv = 20.0 * np.log10(max(rms_v, 1e-30) / 1e-6)

    # Resolved lines that are already within HARMONIC_PROXIMITY_DB of the limit.
    # Counted on the receiver-emulated trace, because that is the quantity a
    # test house would actually compare against the limit line.
    line_idx = _detect_lines(band_trace, band_limit - HARMONIC_PROXIMITY_DB)
    harmonic_count = int(line_idx.size)

    # THD-like score: energy of switching-related content relative to the
    # robust (median) noise floor of the band, in dB. Always >= 0.
    floor_v = float(np.median(band_amp))
    floor_energy = max(floor_v ** 2 * band_amp.size, 1e-30)
    total_energy = float(np.sum(band_amp ** 2))
    thd_score = float(
        10.0 * np.log10(max(total_energy - floor_energy, 0.0) / floor_energy + 1.0)
    )

    # Margin is the *worst* headroom anywhere in the band, not just at the peak,
    # because the limit line is not flat within bands A and C.
    margin_curve = band_limit - band_trace
    worst_idx = int(np.argmin(margin_curve))
    margin_db = float(margin_curve[worst_idx])

    return BandAnalysis(
        key=band.key,
        label=band.label,
        f_low_hz=band.f_low_hz,
        f_high_hz=band.f_high_hz,
        peak_dbuv=peak_dbuv,
        peak_frequency_hz=float(band_freqs[peak_idx]),
        rms_dbuv=rms_dbuv,
        harmonic_count=harmonic_count,
        thd_score=thd_score,
        limit_at_peak_dbuv=float(band_limit[peak_idx]),
        margin_db=margin_db,
        worst_frequency_hz=float(band_freqs[worst_idx]),
        passes=bool(margin_db > 0.0),
    )


def _chart_trace(freqs: np.ndarray, trace_dbuv: np.ndarray) -> SpectrumTrace:
    """Max-hold down-sample onto a log-spaced grid, like a spectrum display."""
    edges = np.logspace(
        np.log10(ANALYSIS_F_LOW_HZ), np.log10(ANALYSIS_F_HIGH_HZ), CHART_POINTS + 1
    )
    bin_index = np.searchsorted(edges, freqs, side="right") - 1
    valid = (bin_index >= 0) & (bin_index < CHART_POINTS)

    held = np.full(CHART_POINTS, -np.inf)
    np.maximum.at(held, bin_index[valid], trace_dbuv[valid])

    centres = np.sqrt(edges[:-1] * edges[1:])
    # A few log bins can be narrower than the FFT bin spacing at the low end;
    # fill them by interpolating from the populated bins.
    populated = np.isfinite(held)
    if not populated.all():
        held = np.interp(
            np.log10(centres), np.log10(centres[populated]), held[populated]
        )

    return SpectrumTrace(
        frequency_hz=centres,
        emission_dbuv=held,
        limit_dbuv=limit_dbuv(centres),
    )


def _harmonic_peaks(
    waveform: np.ndarray,
    sample_rate_hz: float,
    fundamental_hz: float,
    n_orders: int = 40,
    top_k: int = 5,
) -> Tuple[float, float, List[HarmonicPeak]]:
    """THD and the strongest harmonics of a periodic current.

    THD is 100 · sqrt(sum I_h^2) / I_1 for h = 2..n_orders, using a Hann-windowed
    FFT and the bin nearest each integer multiple of the fundamental. Returns
    (thd_percent, fundamental_amplitude, top peaks excluding the DC bin).
    """
    n = waveform.size
    window = np.hanning(n)
    spectrum = np.fft.rfft(waveform * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    amplitude = (2.0 / (n * window.mean())) * np.abs(spectrum)
    bin_hz = sample_rate_hz / n

    def _near(freq: float) -> Tuple[int, float, float]:
        centre = int(round(freq / max(bin_hz, 1e-12)))
        lo = max(1, centre - 1)
        hi = min(amplitude.size, centre + 2)
        local = lo + int(np.argmax(amplitude[lo:hi]))
        return local, float(freqs[local]), float(amplitude[local])

    _, _, fund_amp = _near(fundamental_hz)
    fund_amp = max(fund_amp, 1e-12)

    peaks: List[HarmonicPeak] = []
    harmonic_energy = 0.0
    for order in range(1, n_orders + 1):
        _, freq, amp = _near(order * fundamental_hz)
        if order >= 2:
            harmonic_energy += amp ** 2
        peaks.append(HarmonicPeak(
            order=order,
            frequency_hz=freq,
            amplitude=amp,
            percent_of_fundamental=100.0 * amp / fund_amp,
        ))

    thd = 100.0 * float(np.sqrt(harmonic_energy)) / fund_amp
    ranked = sorted(peaks[1:], key=lambda p: p.amplitude, reverse=True)
    return thd, fund_amp, ranked[:top_k]


def _dominant_statement(label: str, peaks: Sequence[HarmonicPeak]) -> Tuple[Tuple[int, ...], str]:
    if not peaks:
        return (), f"No resolved harmonics on {label}."
    ranked = [p for p in peaks if p.order > 1]
    if not ranked or ranked[0].percent_of_fundamental < 2.0:
        return (), f"{label} is essentially sinusoidal."
    top = [ranked[0]]
    if (
        len(ranked) > 1
        and ranked[1].percent_of_fundamental >= max(1.5, 0.25 * ranked[0].percent_of_fundamental)
    ):
        top.append(ranked[1])
    orders = tuple(p.order for p in top)
    names = " and ".join(_ordinal(o) for o in orders)
    if len(orders) == 1:
        return orders, f"The {names} harmonic dominates {label} distortion."
    return orders, f"{names} harmonics dominate {label} distortion."


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def analyse_power_quality(sim: SimulationResult) -> List[PowerQualityReport]:
    """THD of motor current and input current on the long power-quality record."""
    from simulate import MAINS_HZ

    reports: List[PowerQualityReport] = []
    pairs = (
        ("motor_current", "motor current", sim.pq_motor_current_a),
        ("input_current", "input current", sim.pq_input_current_a),
    )
    fs = sim.pq_sample_rate_hz
    for key, label, waveform in pairs:
        if waveform is None or waveform.size < 32:
            continue
        thd, fund_amp, peaks = _harmonic_peaks(waveform, fs, MAINS_HZ)
        orders, statement = _dominant_statement(label, peaks)
        reports.append(PowerQualityReport(
            key=key,
            label=label,
            thd_percent=float(thd),
            fundamental_hz=MAINS_HZ,
            fundamental_amplitude=float(fund_amp),
            peaks=peaks,
            dominant_orders=orders,
            dominant_statement=statement,
        ))
    return reports


def extract_features(sim: SimulationResult) -> FeatureBundle:
    """Full analysis chain: FFT -> receiver emulation -> band features + margins."""
    freqs, amplitude = _amplitude_spectrum(sim.measured_segments, sim.sample_rate_hz)
    n_samples = sim.measured_segments.shape[1]

    in_range = (freqs >= ANALYSIS_F_LOW_HZ) & (freqs <= ANALYSIS_F_HIGH_HZ)
    freqs = freqs[in_range]
    amplitude = amplitude[in_range]

    trace = _receiver_trace(freqs, amplitude, sim.sample_rate_hz, n_samples)

    bands = [_analyse_band(b, freqs, trace, amplitude) for b in EMC_BANDS]

    params = sim.parameters
    design_values = [getattr(params, key) for key in ML_PARAMETER_KEYS]
    design_values.append(float(params.pwm_cm_penalty_db))

    band_values: List[float] = []
    for band in bands:
        band_values.extend(
            [
                band.peak_dbuv,
                band.rms_dbuv,
                float(band.harmonic_count),
                band.thd_score,
            ]
        )

    feature_vector = np.asarray(design_values + band_values, dtype=float)
    assert feature_vector.size == len(FEATURE_NAMES), "feature schema drift"

    return FeatureBundle(
        bands=bands,
        spectrum=_chart_trace(freqs, trace),
        feature_vector=feature_vector,
        diagnostics=dict(sim.diagnostics),
        power_quality=analyse_power_quality(sim),
    )


# ---------------------------------------------------------------------------
# Label helpers used by train_model.py and app.py
# ---------------------------------------------------------------------------
def band_margins(bundle: FeatureBundle) -> np.ndarray:
    return np.asarray([b.margin_db for b in bundle.bands], dtype=float)


def band_failures(bundle: FeatureBundle) -> np.ndarray:
    """1 = band exceeds the synthetic limit (fails), 0 = compliant."""
    return (band_margins(bundle) <= 0.0).astype(int)


def parameter_risk_position(key: str, value: float) -> float:
    """Where a parameter sits on its own risk axis, in [0, 1].

    0 = safest end of the admissible range, 1 = riskiest end. Uses the
    physics-informed sign so that e.g. high shielding maps to 0.
    """
    if key == "pwm_cm_penalty_db":
        from simulate import PWM_CM_PENALTY_RANGE

        low, high = PWM_CM_PENALTY_RANGE
        return float(np.clip((value - low) / max(high - low, 1e-9), 0.0, 1.0))
    rge = PARAMETER_RANGE_BY_KEY[key]
    normalised = rge.normalise(value)
    sign = next(s.risk_sign for s in DESIGN_FEATURE_SPECS if s.name == key)
    return normalised if sign > 0 else 1.0 - normalised


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from simulate import DeviceParameters, simulate_device

    demo = DeviceParameters(
        switching_frequency_khz=8.0,
        dv_dt_v_per_us=3500.0,
        cable_length_m=25.0,
        shielding_quality=0.6,
        load_current_a=45.0,
        pwm_modulation_type="SPWM",
    )
    bundle = extract_features(simulate_device(demo))
    print(f"{'band':18s} {'peak':>8s} {'limit':>7s} {'margin':>8s} {'lines':>6s} {'thd':>7s}")
    for band in bundle.bands:
        print(
            f"{band.label:18s} {band.peak_dbuv:8.1f} {band.limit_at_peak_dbuv:7.1f} "
            f"{band.margin_db:8.1f} {band.harmonic_count:6d} {band.thd_score:7.1f}"
        )
    print(f"\nfeature vector ({bundle.feature_vector.size} features):")
    for name, value in bundle.as_feature_dict().items():
        print(f"  {name:26s} {value:12.3f}")
