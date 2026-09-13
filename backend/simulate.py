"""
simulate.py -- Physics-based signal simulation layer for CertifAI.

Purpose
-------
Given a small set of *design-time* parameters for an elevator drive / controller
(switching frequency, dv/dt, motor cable length, shielding quality, load current,
PWM modulation strategy), synthesise a realistic time-domain common-mode (CM)
disturbance waveform as it would be seen by a LISN / CM current probe at the
equipment's power port.

The output of this module is consumed by ``features.py``, which performs the FFT
and band analysis.

Physical model
--------------
The chain modelled here is the standard conducted-emission mechanism for a
voltage-source inverter driving a motor over a (possibly shielded) cable:

    1. Three inverter legs switch between the DC-link rails. Each transition is
       modelled as a *trapezoid* whose edge slope is exactly the user-supplied
       dv/dt. This is the classical trapezoidal EMI model: the pulse train sets
       the harmonic *line spacing*, and the edge rise time sets the *envelope
       roll-off* (the second "corner" of the well-known 20/40 dB-per-decade
       trapezoid spectrum, located at 1 / (pi * t_rise)).

    2. The common-mode voltage is v_cm = (v_a + v_b + v_c) / 3. For a two-level
       inverter this is a staircase with V_DC/3 steps that switches six times
       per carrier period. This is the true driver of conducted CM emissions --
       it is *not* the same as the differential (line-to-line) voltage.

    3. The motor cable presents a distributed parasitic capacitance to
       protective earth (C_par = c_per_metre x length). The CM current is
       i_cm = C_par * dv_cm/dt, so emissions rise roughly 20 dB/decade with
       cable length.

    4. The cable also forms a resonant transmission-line structure. Every
       switching edge rings at the cable's quarter-wave resonance,
       f_res ~ v_prop / (4 * length), *and* at its odd harmonics, because an
       open-ended line resonates at f_res, 3 f_res, 5 f_res, ... with mode
       amplitudes falling as 1/n. This is modelled as a direct path plus a bank
       of second-order underdamped resonators. Modelling the higher modes
       matters: with only the fundamental mode, lengthening the cable would slide
       the single resonant peak *out* of the 5-30 MHz band and make emissions
       there fall with length, contradicting the physics that longer cables
       couple more. (Linear operations, so applying them once to the summed CM
       waveform is equivalent to applying them per leg.)

    5. Cable screening plus the CM choke are modelled as (a) a broadband
       insertion loss and (b) a first-order low-pass, because real screens and
       ferrites attenuate far more at high frequency than at 150 kHz.

    6. A randomised noise floor (white + 1/f) is added in the measured-voltage
       domain so that the synthetic spectra have a realistic, non-deterministic
       baseline.

Receiver dwell / segmentation
----------------------------
A single short capture is *not* representative: the inverter's duty cycles vary
over the 20 ms motor fundamental period, and a 0.3 ms window only sees a small
slice of that. Capturing one window makes the measured level swing by several dB
purely as a function of where in the fundamental cycle the capture landed.

A real EMI receiver avoids this by dwelling on each frequency for many
fundamental cycles and max-holding. We emulate that by simulating
``N_SEGMENTS`` short records spaced evenly across one fundamental period;
``features.py`` then max-holds their spectra. This is both faster than one very
long capture and a closer match to how the measurement is actually performed.

All segments are computed as a single batched array operation, so the cost is
essentially one long FFT-free pass plus a handful of vectorised IIR filters.

Monotonicity (this is what makes the downstream ML model "physics-informed")
---------------------------------------------------------------------------
By construction, the *expected* emission level in every band is:

    increasing in  switching_frequency_khz   (more transitions per second; for an
                                              impulsive pulse train the spectral
                                              line amplitude scales with the
                                              pulse repetition frequency)
    increasing in  dv_dt_v_per_us            (faster edges move the trapezoid's
                                              corner frequency up, adding HF
                                              energy without removing any)
    increasing in  cable_length_m            (larger C_par -> larger i_cm)
    decreasing in  shielding_quality         (insertion loss + HF roll-off)
    increasing in  load_current_a            (larger circulating currents)
    decreasing in  PWM_SPREAD_INDEX          (fewer or less-correlated switching
                                              transitions -> lower peak lines)

``train_model.py`` encodes exactly these six signs as XGBoost
``monotone_constraints``, and its validation step empirically re-checks that the
simulator itself obeys them (see ``tools/sweep_check.py``).

IMPORTANT SCOPE NOTE
--------------------
This is a *synthetic* model. It is calibrated to produce levels in the same
decade as real drive measurements (roughly 30-110 dBuV) and to reproduce the
correct qualitative trends, but it is not fitted to measured hardware. Absolute
levels must not be interpreted as predicted lab results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Final, Optional, Tuple

import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# Acquisition grid
# ---------------------------------------------------------------------------
# Nyquist must comfortably exceed the 30 MHz top of the conducted-emission range
# and must resolve edges as fast as 10 kV/us (a 565 V step in ~57 ns -> ~11
# samples per edge at 200 MS/s).
SAMPLE_RATE_HZ: Final[float] = 200e6

# Per-segment record length. 65536 samples at 200 MS/s is a 328 us window with
# 3.05 kHz bin spacing -- comfortably finer than the 9 kHz CISPR resolution
# bandwidth used in features.py, which is the figure that actually governs how
# finely a receiver can resolve adjacent lines.
N_SAMPLES: Final[int] = 2 ** 16

# Number of records spaced across one motor fundamental period (see the
# "Receiver dwell / segmentation" note in the module docstring).
N_SEGMENTS: Final[int] = 6

# Samples discarded at the head of each segment so the resonance and shielding
# IIR filters settle before the record is kept (~10 us, far longer than the
# slowest modelled time constant).
N_WARMUP_SAMPLES: Final[int] = 2048

# ---------------------------------------------------------------------------
# Power-stage constants (fixed; not user-facing design parameters)
# ---------------------------------------------------------------------------
V_DC_LINK: Final[float] = 565.0          # V, 400 V AC three-phase rectified
FUNDAMENTAL_OUT_HZ: Final[float] = 50.0  # motor fundamental at rated travel speed
MODULATION_INDEX: Final[float] = 0.85

CABLE_C_PER_METRE_F: Final[float] = 100e-12   # F/m line-to-earth, typical 4-core motor cable
CABLE_V_PROP_M_PER_S: Final[float] = 2.0e8    # ~0.67c in PVC-insulated cable
CABLE_RESONANCE_Q: Final[float] = 3.0         # held constant so length only shifts the modes
CABLE_RESONANCE_MODES: Final[int] = 3         # quarter-wave mode plus 3f, 5f harmonics
CABLE_RESONANCE_GAIN: Final[float] = 1.5      # weight of the resonant path vs the direct path
LISN_IMPEDANCE_OHM: Final[float] = 50.0

# Reference load current for the sub-linear amplitude scaling. Emissions grow
# with load current but less than proportionally, because much of the CM path is
# displacement current driven by dv/dt rather than by load current.
LOAD_CURRENT_REF_A: Final[float] = 40.0
LOAD_CURRENT_EXPONENT: Final[float] = 0.5

# Shielding model. shielding_quality = 0 -> unscreened cable, no CM choke.
# shielding_quality = 1 -> well-terminated screen plus a CM choke.
SHIELD_MAX_INSERTION_LOSS_DB: Final[float] = 26.0
SHIELD_LOWPASS_MAX_HZ: Final[float] = 60e6   # cutoff with no shielding (i.e. no roll-off)
SHIELD_LOWPASS_MIN_HZ: Final[float] = 3.0e6  # cutoff with perfect shielding

# Dimensionless calibration constant folding together the CM return-path
# impedance divider, probe transfer impedance and LISN loading.
#
# ASSUMPTION: this constant is *chosen*, not measured. It is set so that a
# mid-range configuration lands close to the assumed limit line, which is what
# makes the pass/fail decision boundary informative across the parameter space.
# Tuned with tools/sweep_check.py; see that script for the resulting margin
# distribution across the parameter space.
COUPLING_CALIBRATION: Final[float] = 5.0e-3

# Randomised noise floor, expressed as a level *per FFT bin* in dBuV.
#
# Kept deliberately low and narrow. After 9 kHz RBW integration, max-hold across
# dwell segments, and taking the maximum across the thousands of bins in a band,
# this presents as roughly a 25-40 dBuV displayed floor -- realistic for a
# conducted-emission setup, and low enough that the physics (not the noise draw)
# governs the band levels.
NOISE_FLOOR_DBUV_RANGE: Final[Tuple[float, float]] = (2.0, 12.0)
NOISE_PINK_FRACTION: Final[float] = 0.45  # share of noise power that is 1/f

# ---------------------------------------------------------------------------
# Design parameter space
# ---------------------------------------------------------------------------
# Presentation order for the UI (conventional -> more specialised).
PWM_MODULATION_TYPES: Final[Tuple[str, ...]] = (
    "SPWM",
    "SVPWM",
    "DPWM",
    "RANDOM_SPWM",
)

PWM_MODULATION_LABELS: Final[Dict[str, str]] = {
    "SPWM": "Sinusoidal PWM",
    "SVPWM": "Space-Vector PWM",
    "DPWM": "Discontinuous PWM",
    "RANDOM_SPWM": "Randomised-Carrier PWM",
}

# Modelled change in common-mode emission relative to plain SPWM, in dB.
#
# This is the model's encoding of the modulation strategy. It is a *physical
# quantity in dB* rather than an arbitrary rank, which matters for two reasons:
# the downstream monotone constraint then applies to something meaningful, and
# two strategies the simulator genuinely cannot distinguish can be given the same
# value instead of a fabricated ordering.
#
# Values are calibrated against the simulator itself, averaged over 150 random
# designs per strategy (``tools/sweep_check.py --pwm``). Mechanisms:
#
#   SVPWM        +1.0  Same commutation count as SPWM, but the min-max
#                      zero-sequence injection deliberately adds a common-mode
#                      component to the reference. Common mode is exactly what
#                      couples to earth, so SVPWM is marginally the worst choice
#                      for conducted CM emissions -- the known trade-off against
#                      its better DC-link utilisation.
#   SPWM          0.0  Reference case. All three legs commutate every carrier
#                      period.
#   RANDOM_SPWM  +1.0  Same commutation count as SPWM, with a jittered carrier
#                      period. Note the sign: in this model carrier randomisation
#                      makes the *measured* level slightly worse, not better. See
#                      LIMITATION 2 below.
#   DPWM         -2.5  Discontinuous modulation clamps one leg to a DC rail at all
#                      times, removing about a third of all commutations. Spectral
#                      line height scales with repetition rate, so fewer current
#                      impulses per second means a lower measured level. The ideal
#                      figure would be 20*log10(2/3) = -3.5 dB; the simulator
#                      measures -2.5 dB, which is the value used.
#
# LIMITATION 1: real drives sometimes show DPWM performing *worse* than SPWM at
# low frequencies because of increased low-order zero-sequence content. That
# content sits at multiples of 150 Hz, three orders of magnitude below the
# 150 kHz measurement floor, so this model does not reproduce it. Within the
# 150 kHz-30 MHz range modelled here, commutation count dominates.
#
# LIMITATION 2: randomised-carrier PWM comes out ~1 dB *worse* here, against the
# 5-10 dB improvement often quoted. That is a property of the measurement rather
# than a bug, and it is worth understanding before dismissing it:
#
#   * A 2-20 kHz carrier produces harmonics spaced 2-20 kHz apart, at or below
#     the 9 kHz CISPR resolution bandwidth. The receiver already cannot resolve
#     individual lines in this range, so there is no line structure for spreading
#     to break up.
#   * Randomisation does not reduce total emitted energy. Against a peak /
#     max-hold detector observing what is effectively a continuum, the extra
#     run-to-run variance simply gives the detector more high excursions to catch.
#
# Carrier randomisation does pay off for high-carrier (100 kHz+ SiC) designs where
# lines are resolved, but those sit outside this tool's parameter range.
PWM_CM_PENALTY_DB: Final[Dict[str, float]] = {
    "SVPWM": 1.0,
    "RANDOM_SPWM": 1.0,
    "SPWM": 0.0,
    "DPWM": -2.5,
}

PWM_CM_PENALTY_RANGE: Final[Tuple[float, float]] = (
    min(PWM_CM_PENALTY_DB.values()),
    max(PWM_CM_PENALTY_DB.values()),
)

RANDOM_SPWM_JITTER: Final[float] = 0.35  # +/-35 % per-period carrier jitter


@dataclass(frozen=True)
class ParameterRange:
    """Admissible range and presentation metadata for one design parameter."""

    key: str
    label: str
    unit: str
    minimum: float
    maximum: float
    step: float
    default: float
    description: str

    def clamp(self, value: float) -> float:
        return float(np.clip(value, self.minimum, self.maximum))

    def normalise(self, value: float) -> float:
        """Map into [0, 1] across the admissible range."""
        span = self.maximum - self.minimum
        if span <= 0:
            return 0.0
        return float(np.clip((value - self.minimum) / span, 0.0, 1.0))


PARAMETER_RANGES: Final[Tuple[ParameterRange, ...]] = (
    ParameterRange(
        key="switching_frequency_khz",
        label="Switching Frequency",
        unit="kHz",
        minimum=2.0,
        maximum=20.0,
        step=0.1,
        default=8.0,
        description=(
            "IGBT/SiC carrier frequency. Higher carriers give smoother motor "
            "current and quieter acoustics, but place more switching "
            "transitions per second into the 150 kHz-30 MHz measurement range."
        ),
    ),
    ParameterRange(
        key="dv_dt_v_per_us",
        label="Switching dv/dt",
        unit="V/us",
        minimum=500.0,
        maximum=10000.0,
        step=50.0,
        default=3500.0,
        description=(
            "Slew rate of each switching edge at the inverter output. Faster "
            "edges cut switching loss but push the spectral envelope corner "
            "higher, broadening high-frequency emissions."
        ),
    ),
    ParameterRange(
        key="cable_length_m",
        label="Motor Cable Length",
        unit="m",
        minimum=1.0,
        maximum=120.0,
        step=1.0,
        default=25.0,
        description=(
            "Length of the drive-to-machine cable in the shaft. Longer runs add "
            "parasitic capacitance to earth (more common-mode current) and lower "
            "the cable's resonant frequency."
        ),
    ),
    ParameterRange(
        key="shielding_quality",
        label="Shielding Quality",
        unit="",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        default=0.6,
        description=(
            "Combined effectiveness of cable screening, 360-degree screen "
            "termination and any common-mode choke. 0 = unscreened cable, "
            "1 = fully terminated screen plus CM filtering."
        ),
    ),
    ParameterRange(
        key="load_current_a",
        label="Load Current",
        unit="A",
        minimum=5.0,
        maximum=200.0,
        step=1.0,
        default=45.0,
        description=(
            "RMS motor current at the assessed duty point. Larger currents "
            "raise circulating and return-path currents, lifting emission "
            "amplitude across all bands."
        ),
    ),
)

PARAMETER_RANGE_BY_KEY: Final[Dict[str, ParameterRange]] = {
    r.key: r for r in PARAMETER_RANGES
}

NUMERIC_PARAMETER_KEYS: Final[Tuple[str, ...]] = tuple(
    r.key for r in PARAMETER_RANGES
)


@dataclass(frozen=True)
class DeviceParameters:
    """The six design-time inputs that fully determine a simulation."""

    switching_frequency_khz: float
    dv_dt_v_per_us: float
    cable_length_m: float
    shielding_quality: float
    load_current_a: float
    pwm_modulation_type: str = "SPWM"

    def __post_init__(self) -> None:
        if self.pwm_modulation_type not in PWM_CM_PENALTY_DB:
            raise ValueError(
                f"pwm_modulation_type must be one of {PWM_MODULATION_TYPES}, "
                f"got {self.pwm_modulation_type!r}"
            )

    @classmethod
    def clamped(cls, **kwargs: object) -> "DeviceParameters":
        """Build an instance with numeric fields clamped to admissible ranges."""
        values: Dict[str, object] = {}
        for key in NUMERIC_PARAMETER_KEYS:
            values[key] = PARAMETER_RANGE_BY_KEY[key].clamp(float(kwargs[key]))  # type: ignore[arg-type]
        values["pwm_modulation_type"] = kwargs.get("pwm_modulation_type", "SPWM")
        return cls(**values)  # type: ignore[arg-type]

    def as_dict(self) -> Dict[str, object]:
        return {
            "switching_frequency_khz": self.switching_frequency_khz,
            "dv_dt_v_per_us": self.dv_dt_v_per_us,
            "cable_length_m": self.cable_length_m,
            "shielding_quality": self.shielding_quality,
            "load_current_a": self.load_current_a,
            "pwm_modulation_type": self.pwm_modulation_type,
        }

    @property
    def pwm_cm_penalty_db(self) -> float:
        """Modelled CM-emission offset of this strategy relative to SPWM, in dB."""
        return PWM_CM_PENALTY_DB[self.pwm_modulation_type]

    def deterministic_seed(self) -> int:
        """Stable seed so identical parameters always give an identical report."""
        payload = "|".join(
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(self.as_dict().items())
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")


@dataclass
class SimulationResult:
    """Time-domain output of one simulation run.

    ``measured_segments`` has shape ``(N_SEGMENTS, N_SAMPLES)``; each row is one
    receiver dwell window at a different phase of the motor fundamental.
    ``time_s`` and ``cm_voltage_v`` describe the first segment only and exist for
    time-domain inspection.
    """

    time_s: np.ndarray
    cm_voltage_v: np.ndarray            # inverter common-mode voltage (V), segment 0
    measured_segments: np.ndarray       # disturbance voltage at the LISN (V)
    sample_rate_hz: float
    parameters: DeviceParameters
    diagnostics: Dict[str, float] = field(default_factory=dict)

    @property
    def measured_voltage_v(self) -> np.ndarray:
        """First dwell window, for time-domain plots and smoke checks."""
        return self.measured_segments[0]


# ---------------------------------------------------------------------------
# PWM reference / carrier generation
# ---------------------------------------------------------------------------
def _carrier_phase(
    t: np.ndarray,
    dt: float,
    carrier_hz: float,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Instantaneous carrier phase in [0, 1) for a batch of time rows.

    ``t`` has shape ``(n_segments, n_samples)``. Carrier jitter is what makes
    randomised PWM a spread-spectrum technique: the switching energy is smeared
    across sidebands instead of concentrating on exact multiples of the carrier.
    """
    if jitter <= 0.0:
        return (t * carrier_hz) % 1.0

    # Piecewise-constant frequency, re-drawn once per nominal carrier period and
    # independently per segment.
    n_segments, n_samples = t.shape
    samples_per_period = max(2, int(round(1.0 / (carrier_hz * dt))))
    n_periods = int(np.ceil(n_samples / samples_per_period)) + 1
    factors = 1.0 + rng.uniform(-jitter, jitter, size=(n_segments, n_periods))
    inst_freq = np.repeat(carrier_hz * factors, samples_per_period, axis=1)[:, :n_samples]
    return (t[:, :1] * carrier_hz + np.cumsum(inst_freq, axis=1) * dt) % 1.0


def _triangle_from_phase(phase: np.ndarray) -> np.ndarray:
    """Symmetric triangular carrier in [-1, 1] from a phase in [0, 1)."""
    return 2.0 * np.abs(2.0 * (phase - 0.5)) - 1.0


def _modulation_references(t: np.ndarray, modulation: str) -> np.ndarray:
    """Three-phase references in [-1, 1], shape ``(n_segments, 3, n_samples)``."""
    omega = 2.0 * np.pi * FUNDAMENTAL_OUT_HZ
    leg_phases = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
    ref = MODULATION_INDEX * np.sin(
        omega * t[:, None, :] + leg_phases[None, :, None]
    )

    if modulation == "SVPWM":
        # Min-max (zero-sequence) injection: the standard continuous SVPWM
        # equivalent. It extends the linear modulation range but deliberately
        # injects a common-mode component, which is what couples to earth.
        zero_seq = -0.5 * (ref.max(axis=1) + ref.min(axis=1))
        ref = ref + zero_seq[:, None, :]
    elif modulation == "DPWM":
        # Discontinuous PWM: at every instant the phase with the largest
        # magnitude is clamped to a DC rail, so that leg stops commutating.
        dominant = np.argmax(np.abs(ref), axis=1, keepdims=True)
        dominant_value = np.take_along_axis(ref, dominant, axis=1)
        ref = ref + (np.sign(dominant_value) - dominant_value)
    # "SPWM" and "RANDOM_SPWM" use the plain sinusoidal reference; the latter
    # differs only in its carrier (handled in _carrier_phase).

    return np.clip(ref, -1.0, 1.0)


def _switching_states(
    params: DeviceParameters, t: np.ndarray, dt: float, rng: np.random.Generator
) -> np.ndarray:
    """Ideal (+1 / -1) leg states, shape ``(n_segments, 3, n_samples)``."""
    carrier_hz = params.switching_frequency_khz * 1e3
    jitter = (
        RANDOM_SPWM_JITTER if params.pwm_modulation_type == "RANDOM_SPWM" else 0.0
    )
    carrier = _triangle_from_phase(_carrier_phase(t, dt, carrier_hz, jitter, rng))
    ref = _modulation_references(t, params.pwm_modulation_type)
    return np.where(ref > carrier[:, None, :], 1.0, -1.0)


# ---------------------------------------------------------------------------
# Trapezoidal edge shaping
# ---------------------------------------------------------------------------
def _causal_box_filter(x: np.ndarray, width: int) -> np.ndarray:
    """Normalised causal moving average along the last axis, in O(N).

    Equivalent to ``convolve(x, ones(width) / width)`` truncated to the input
    length, but without the O(N * width) cost -- which matters because the slew
    window is ~226 samples at the slowest modelled dv/dt.
    """
    if width <= 1:
        return x
    cumulative = np.cumsum(x, axis=-1)
    shifted = np.zeros_like(cumulative)
    shifted[..., width:] = cumulative[..., :-width]
    return (cumulative - shifted) / width


def _common_mode_voltage(
    states: np.ndarray, params: DeviceParameters, dt: float
) -> np.ndarray:
    """Trapezoidal leg voltages averaged into the common-mode voltage.

    Rather than rendering each edge individually, we build the *derivative*
    signal (a train of rectangular slew pulses of height dv/dt) and integrate
    it. This is exact for a trapezoid and is fully vectorised over segments and
    legs.

    Returns shape ``(n_segments, n_samples)`` in volts.
    """
    rise_time_s = V_DC_LINK / (params.dv_dt_v_per_us * 1e6)
    n_rise = max(1, int(round(rise_time_s / dt)))

    # Signed impulses at each transition, in units of "steps of V_DC".
    transitions = np.zeros_like(states)
    transitions[..., 1:] = np.diff(states, axis=-1) / 2.0

    # Spread each impulse over the slew window, then integrate to get voltage.
    spread = _causal_box_filter(transitions, n_rise)
    legs = (np.cumsum(spread, axis=-1) - 0.5) * V_DC_LINK
    return legs.mean(axis=1)


def _apply_cable_resonance(
    v_cm: np.ndarray, params: DeviceParameters, sample_rate_hz: float
) -> Tuple[np.ndarray, float]:
    """Ring the CM waveform on the motor cable's transmission-line modes.

    An open-ended line resonates at its quarter-wave frequency and at the odd
    harmonics of it, with mode amplitudes falling roughly as 1/n. We model a
    direct (non-resonant) path in parallel with a bank of ``CABLE_RESONANCE_MODES``
    second-order underdamped resonators at f_res, 3 f_res, 5 f_res.

    The direct path is what keeps the length -> emission trend monotonic: a
    resonator bank on its own would slide all of its peaks below 5 MHz for a long
    cable and make high-band emissions *fall* with length. With a direct path
    present, lengthening the cable always raises the level (through C_par) and
    the modes only add a bounded bump wherever they happen to land.

    Q is held constant so cable length only shifts the mode frequencies.
    """
    f_res = CABLE_V_PROP_M_PER_S / (4.0 * max(params.cable_length_m, 0.5))
    f_res = float(np.clip(f_res, 0.4e6, 0.40 * sample_rate_hz))

    zeta = 1.0 / (2.0 * CABLE_RESONANCE_Q)
    nyquist = 0.5 * sample_rate_hz

    resonant = np.zeros_like(v_cm)
    weight_total = 0.0
    for mode in range(CABLE_RESONANCE_MODES):
        order = 2 * mode + 1                 # 1st, 3rd, 5th ... quarter-wave mode
        f_mode = f_res * order
        if f_mode >= 0.8 * nyquist:
            break
        weight = 1.0 / order                 # transmission-line mode amplitudes ~ 1/n
        wn = 2.0 * np.pi * f_mode
        b, a = signal.bilinear(
            [wn ** 2], [1.0, 2.0 * zeta * wn, wn ** 2], fs=sample_rate_hz
        )
        resonant += weight * signal.lfilter(b, a, v_cm, axis=-1)
        weight_total += weight

    if weight_total > 0.0:
        resonant /= weight_total

    return v_cm + CABLE_RESONANCE_GAIN * resonant, f_res


def _shielding_filter(
    params: DeviceParameters, sample_rate_hz: float
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Screen + CM-choke model: broadband insertion loss plus HF roll-off."""
    q = float(np.clip(params.shielding_quality, 0.0, 1.0))

    insertion_loss_db = SHIELD_MAX_INSERTION_LOSS_DB * (q ** 0.85)
    broadband_gain = 10.0 ** (-insertion_loss_db / 20.0)

    # Log-interpolated cutoff: real screens and ferrites are far more effective
    # at tens of MHz than at 150 kHz, so shielding acts as a low-pass.
    log_hi, log_lo = np.log10(SHIELD_LOWPASS_MAX_HZ), np.log10(SHIELD_LOWPASS_MIN_HZ)
    cutoff_hz = 10.0 ** (log_hi + q * (log_lo - log_hi))
    wn = float(np.clip(cutoff_hz / (0.5 * sample_rate_hz), 1e-4, 0.98))
    b, a = signal.butter(1, wn, btype="low")

    return b, a, broadband_gain, insertion_loss_db, cutoff_hz


def _noise_floor(
    shape: Tuple[int, int], target_dbuv: float, rng: np.random.Generator
) -> np.ndarray:
    """White + 1/f noise scaled to a given per-FFT-bin level in dBuV.

    For real Gaussian noise of standard deviation sigma, the single-sided
    amplitude line computed as 2|X_k|/N has expectation sigma * sqrt(pi / N).
    We invert that relation so the requested floor is hit directly.
    """
    n_samples = shape[1]
    target_v = 10.0 ** (target_dbuv / 20.0) * 1e-6
    sigma = target_v * np.sqrt(n_samples / np.pi)

    white = rng.standard_normal(shape)
    # One-pole integration of the same white sequence approximates a 1/f slope.
    # Reusing the sequence correlates the two components, which is harmless here
    # (we only care about the resulting spectral shape) and halves the RNG cost.
    pink = signal.lfilter([1.0], [1.0, -0.995], white, axis=-1)
    pink /= np.std(pink, axis=-1, keepdims=True) + 1e-30

    mix = (
        np.sqrt(1.0 - NOISE_PINK_FRACTION) * white
        + np.sqrt(NOISE_PINK_FRACTION) * pink
    )
    mix /= np.std(mix, axis=-1, keepdims=True) + 1e-30
    return sigma * mix


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def simulate_device(
    params: DeviceParameters,
    *,
    seed: Optional[int] = None,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    n_samples: int = N_SAMPLES,
    n_segments: int = N_SEGMENTS,
) -> SimulationResult:
    """Simulate the conducted common-mode disturbance for one device design.

    Parameters
    ----------
    params:
        The six design-time inputs.
    seed:
        Seed for the randomised noise floor and carrier jitter. Defaults to a
        hash of ``params`` so that the same design always yields the same
        report -- an assessment tool must be reproducible.

    Returns
    -------
    SimulationResult holding one dwell window per fundamental phase.
    """
    rng = np.random.default_rng(
        params.deterministic_seed() if seed is None else seed
    )
    dt = 1.0 / sample_rate_hz
    total = n_samples + N_WARMUP_SAMPLES

    # Dwell windows spaced evenly across one motor fundamental period, so the
    # full range of inverter duty cycles is represented.
    segment_starts = (
        np.arange(n_segments) / (FUNDAMENTAL_OUT_HZ * n_segments)
    )[:, None]
    t = segment_starts + np.arange(total)[None, :] * dt

    # 1-2. Trapezoidal leg voltages -> common-mode voltage.
    states = _switching_states(params, t, dt, rng)
    v_cm = _common_mode_voltage(states, params, dt)

    # 4. Cable resonance / edge ringing.
    v_cm, f_res = _apply_cable_resonance(v_cm, params, sample_rate_hz)

    # 3. Displacement current into earth through the cable's parasitic C, scaled
    #    by the sub-linear load-current dependence.
    c_par_f = CABLE_C_PER_METRE_F * params.cable_length_m
    load_gain = (params.load_current_a / LOAD_CURRENT_REF_A) ** LOAD_CURRENT_EXPONENT
    i_cm = c_par_f * np.gradient(v_cm, dt, axis=-1) * load_gain

    # 5. Screening and CM choke.
    sh_b, sh_a, broadband_gain, insertion_loss_db, shield_cutoff_hz = (
        _shielding_filter(params, sample_rate_hz)
    )
    i_cm = signal.lfilter(sh_b, sh_a, i_cm, axis=-1) * broadband_gain

    # Disturbance voltage developed across the measuring impedance, plus a
    # randomised instrument / environment noise floor (step 6).
    noise_floor_dbuv = float(rng.uniform(*NOISE_FLOOR_DBUV_RANGE))
    v_meas = i_cm * LISN_IMPEDANCE_OHM * COUPLING_CALIBRATION
    v_meas = v_meas + _noise_floor((n_segments, total), noise_floor_dbuv, rng)

    return SimulationResult(
        time_s=np.arange(n_samples) * dt,
        cm_voltage_v=v_cm[0, N_WARMUP_SAMPLES:],
        measured_segments=np.ascontiguousarray(v_meas[:, N_WARMUP_SAMPLES:]),
        sample_rate_hz=sample_rate_hz,
        parameters=params,
        diagnostics={
            "n_segments": float(n_segments),
            "rise_time_ns": V_DC_LINK / (params.dv_dt_v_per_us * 1e6) * 1e9,
            "envelope_corner_hz": params.dv_dt_v_per_us * 1e6 / (np.pi * V_DC_LINK),
            "cable_resonance_hz": f_res,
            "cable_capacitance_pf": c_par_f * 1e12,
            "shield_insertion_loss_db": insertion_loss_db,
            "shield_cutoff_hz": shield_cutoff_hz,
            "noise_floor_dbuv": noise_floor_dbuv,
            "load_gain": load_gain,
        },
    )


def sample_random_parameters(rng: np.random.Generator) -> DeviceParameters:
    """Draw one design uniformly from the admissible parameter space.

    Cable length, dv/dt and load current are drawn log-uniformly because their
    physical effect is per-decade; sampling them linearly would leave the low end
    of the space almost unexplored.
    """
    def _log_uniform(key: str) -> float:
        rge = PARAMETER_RANGE_BY_KEY[key]
        return float(10.0 ** rng.uniform(np.log10(rge.minimum), np.log10(rge.maximum)))

    def _uniform(key: str) -> float:
        rge = PARAMETER_RANGE_BY_KEY[key]
        return float(rng.uniform(rge.minimum, rge.maximum))

    return DeviceParameters(
        switching_frequency_khz=_uniform("switching_frequency_khz"),
        dv_dt_v_per_us=_log_uniform("dv_dt_v_per_us"),
        cable_length_m=_log_uniform("cable_length_m"),
        shielding_quality=_uniform("shielding_quality"),
        load_current_a=_log_uniform("load_current_a"),
        pwm_modulation_type=str(rng.choice(PWM_MODULATION_TYPES)),
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import time

    demo = DeviceParameters(
        switching_frequency_khz=8.0,
        dv_dt_v_per_us=3500.0,
        cable_length_m=25.0,
        shielding_quality=0.6,
        load_current_a=45.0,
        pwm_modulation_type="SPWM",
    )
    t0 = time.perf_counter()
    result = simulate_device(demo)
    elapsed = time.perf_counter() - t0
    peak_dbuv = 20.0 * np.log10(np.abs(result.measured_voltage_v).max() / 1e-6)
    print(f"simulate_device     : {elapsed * 1e3:.0f} ms")
    print(f"segments            : {result.measured_segments.shape}")
    print(f"segment length      : {result.time_s[-1] * 1e6:.1f} us")
    print(f"time-domain peak    : {peak_dbuv:.1f} dBuV")
    for key, value in result.diagnostics.items():
        print(f"{key:26s}: {value:,.3f}")
