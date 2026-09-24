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

    7. Alongside the common-mode EMC path, the same switching states drive four
       additional power-stage waveforms used for the signal explorer and for
       power-quality (THD) analysis: DC-link voltage, motor line-to-line voltage,
       motor current (RL-filtered PWM), and rectifier input current. These are
       *not* fed to the EMC risk models -- they answer a different question.

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

# Common-mode choke, separate from shielding_quality.
# Ott, Electromagnetic Compatibility Engineering (Wiley, 2009), filter chapter:
# a series CM inductance into a resistive termination is a first-order low-pass,
# |Vout/Vin| = 1 / sqrt(1 + (f/fc)^2), fc = Z / (2 π L).
# L = effectiveness * 2 mH into the 50 ohm LISN. 2 mH is a typical power-entry
# CM choke (order of 1 mH). effectiveness 0 leaves the current unchanged.
CM_CHOKE_L_MAX_H: Final[float] = 2.0e-3
CM_CHOKE_Z_OHM: Final[float] = LISN_IMPEDANCE_OHM

# Film capacitor that actually rings with the bus-bar inductance. The 2200 uF
# electrolytic sets the 300 Hz ripple; it is too large to set the HF resonance.
# Skibinski, Kerkman and Schlegel, "EMI Emissions of Modern PWM AC Drives"
# (IEEE Industry Applications Magazine, 1999): the DC bus rings after each
# commutation at the parasitic L and the local film/snubber C.
BUS_FILM_C_F: Final[float] = 4.7e-6
# A real Y-capacitor is not a pure C. Lead and body inductance put the
# self-resonance in the low tens of MHz for a few tens of nanofarads, and a
# couple of ohms of ESR keep the notch finite. Ott treats the ideal shunt as
# a low-pass; these two parasitics stop that low-pass from falling forever.
Y_CAP_ESL_H: Final[float] = 15.0e-9
Y_CAP_ESR_OHM: Final[float] = 2.0
# +/-10% carrier dither. Smaller than RANDOM_SPWM's +/-35%, and in the range
# used for conducted-EMI spread-spectrum (a few to about ten percent).
SPREAD_SPECTRUM_JITTER: Final[float] = 0.10
RECTIFIER_TYPES: Final[Tuple[str, ...]] = ("DIODE_6PULSE", "ACTIVE_FRONT_END")

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

# ---------------------------------------------------------------------------
# Power-quality / multi-signal constants
# ---------------------------------------------------------------------------
# The EMC capture is 328 us at 200 MS/s -- too short to resolve 50 Hz THD.
# A second, cheaper record at 200 kS/s over ~82 ms covers four fundamental
# cycles (12 Hz bins) and still resolves a 16 kHz carrier and its sidebands.
PQ_SAMPLE_RATE_HZ: Final[float] = 200e3
PQ_N_SAMPLES: Final[int] = 2 ** 14
EXPLORER_POINTS: Final[int] = 420
MAINS_HZ: Final[float] = 50.0

# Typical PM elevator-motor stator: a few millihenries. Held constant so that
# current ripple falls with switching frequency exactly as L di/dt = v predicts,
# rather than being confounded with a load-dependent inductance.
MOTOR_L_H: Final[float] = 6.0e-3
MOTOR_R_OHM: Final[float] = 0.45

DC_LINK_CAP_F: Final[float] = 2200e-6
# Maximum AC-line reactor used when input_filter_quality = 1.
INPUT_FILTER_L_MAX_H: Final[float] = 3.5e-3
# 6-pulse rectifier DC current corresponding to the motor RMS current.
RECTIFIER_ID_SCALE: Final[float] = 1.35

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

# Switching-device edge-speed multipliers, applied to the nominal dv/dt and
# di/dt inside the simulators. They are NOT conducted-model features.
#
# Si IGBT is the baseline (x1). SiC MOSFET and GaN HEMT edges are faster at the
# same nominal gate-drive setting because the device capacitances and switching
# energies are smaller:
#   * SiC is commonly about 2-4x a same-class Si IGBT in dv/dt
#     (Wolfspeed CPWR-AN08 and Infineon SiC MOSFET application notes quote
#     Si IGBT edges of roughly 5-15 V/ns against SiC edges of 20-50 V/ns).
#     The factor used here is 2.5, inside that published span.
#   * GaN is commonly about 5-10x a Si IGBT (EPC "GaN FET Switching" and
#     Transphorm application notes quote 50-100 V/ns). The factor used here
#     is 5.0, inside that published span.
SWITCHING_DEVICE_TYPES: Final[Tuple[str, ...]] = (
    "SI_IGBT",
    "SIC_MOSFET",
    "GAN",
)
SWITCHING_DEVICE_EDGE_MULTIPLIER: Final[Dict[str, float]] = {
    "SI_IGBT": 1.0,
    "SIC_MOSFET": 2.5,
    "GAN": 5.0,
}
SWITCHING_DEVICE_LABELS: Final[Dict[str, str]] = {
    "SI_IGBT": "Si IGBT",
    "SIC_MOSFET": "SiC MOSFET",
    "GAN": "GaN",
}

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
        minimum=3.0,
        maximum=16.0,
        step=0.1,
        default=8.0,
        description=(
            "IGBT/SiC carrier frequency, narrowed to the 3-16 kHz range used "
            "in standard elevator VFDs. Higher carriers give smoother motor "
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
            "Cable screening and 360-degree screen termination. 0 = unscreened "
            "cable, 1 = a fully terminated screen. The common-mode choke is a "
            "separate control."
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
            "amplitude across all bands, and increase input-current THD."
        ),
    ),
    ParameterRange(
        key="input_filter_quality",
        label="Input Filter",
        unit="",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        default=0.45,
        description=(
            "Effectiveness of the AC line reactor / DC choke on the rectifier "
            "input. Improves input-current THD (5th and 7th harmonics) but does "
            "not change motor-cable conducted emissions, so it is excluded from "
            "the EMC risk models."
        ),
    ),
    ParameterRange(
        key="cm_choke_effectiveness",
        label="Common-Mode Choke Effectiveness",
        unit="",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        default=0.0,
        description=(
            "A ferrite or toroid on the motor cable that directly suppresses "
            "common-mode noise current — the same noise mechanism this tool's "
            "conducted-emissions score is built around. 0 = no choke. 1 = a "
            "2 mH choke into the 50 ohm LISN, a first-order low-pass "
            "(Ott, Electromagnetic Compatibility Engineering)."
        ),
    ),
    ParameterRange(
        key="clock_frequency_mhz",
        label="Clock Frequency",
        unit="MHz",
        minimum=20.0,
        maximum=200.0,
        step=1.0,
        default=48.0,
        description=(
            "Digital clock on the control board. It does not enter the conducted "
            "common-mode model. It is a source for the separate radiated model."
        ),
    ),
    ParameterRange(
        key="di_dt_a_per_us",
        label="di/dt",
        unit="A/us",
        minimum=20.0,
        maximum=2000.0,
        step=10.0,
        default=200.0,
        description=(
            "Current turn-off rate. Higher di/dt raises radiated magnetic-field "
            "emissions from the cable loop. It does not enter the conducted "
            "common-mode score. Mohan, Undeland and Robbins (Power Electronics, "
            "3rd ed.) treat di/dt as I/t_fall; Erickson and Maksimovic bound it "
            "by the commutation-loop inductance (v = L di/dt). 20-2000 A/us "
            "covers a few tens of amps in about a microsecond down to a fast "
            "module edge."
        ),
    ),
)

# Design parameters that enter the EMC XGBoost models. input_filter_quality is
# a power-quality lever only -- folding it into the EMC feature vector would
# silently invalidate the trained artifacts without changing conducted-emission
# physics.
ML_PARAMETER_KEYS: Final[Tuple[str, ...]] = (
    "switching_frequency_khz",
    "dv_dt_v_per_us",
    "cable_length_m",
    "shielding_quality",
    "load_current_a",
)

PARAMETER_RANGE_BY_KEY: Final[Dict[str, ParameterRange]] = {
    r.key: r for r in PARAMETER_RANGES
}

NUMERIC_PARAMETER_KEYS: Final[Tuple[str, ...]] = tuple(
    r.key for r in PARAMETER_RANGES
)


@dataclass(frozen=True)
class DeviceParameters:
    """The design-time inputs that fully determine a simulation."""

    switching_frequency_khz: float
    dv_dt_v_per_us: float
    cable_length_m: float
    shielding_quality: float
    load_current_a: float
    pwm_modulation_type: str = "SPWM"
    input_filter_quality: float = 0.45
    cm_choke_effectiveness: float = 0.0
    clock_frequency_mhz: float = 48.0
    di_dt_a_per_us: float = 200.0
    switching_device_type: str = "SI_IGBT"
    # Advanced physics inputs. Defaults reproduce the previous simulator exactly.
    # They are not design-only model features and must not be added to SHAP.
    dc_bus_voltage_v: float = V_DC_LINK
    dc_bus_esl_h: float = 0.0
    dc_bus_esr_ohm: float = 0.0
    dead_time_us: float = 0.0
    spread_spectrum: bool = False
    spread_spectrum_jitter: float = SPREAD_SPECTRUM_JITTER
    rectifier_type: str = "DIODE_6PULSE"
    dc_link_choke_h: float = 0.0
    y_capacitance_f: float = 0.0

    def __post_init__(self) -> None:
        if self.pwm_modulation_type not in PWM_CM_PENALTY_DB:
            raise ValueError(
                f"pwm_modulation_type must be one of {PWM_MODULATION_TYPES}, "
                f"got {self.pwm_modulation_type!r}"
            )
        if self.switching_device_type not in SWITCHING_DEVICE_EDGE_MULTIPLIER:
            raise ValueError(
                f"switching_device_type must be one of {SWITCHING_DEVICE_TYPES}, "
                f"got {self.switching_device_type!r}"
            )
        if self.rectifier_type not in RECTIFIER_TYPES:
            raise ValueError(
                f"rectifier_type must be one of {RECTIFIER_TYPES}, "
                f"got {self.rectifier_type!r}"
            )

    @classmethod
    def clamped(cls, **kwargs: object) -> "DeviceParameters":
        """Build an instance with numeric fields clamped to admissible ranges."""
        values: Dict[str, object] = {}
        for key in NUMERIC_PARAMETER_KEYS:
            if key not in kwargs:
                values[key] = PARAMETER_RANGE_BY_KEY[key].default
            else:
                values[key] = PARAMETER_RANGE_BY_KEY[key].clamp(float(kwargs[key]))  # type: ignore[arg-type]
        values["pwm_modulation_type"] = kwargs.get("pwm_modulation_type", "SPWM")
        values["switching_device_type"] = kwargs.get("switching_device_type", "SI_IGBT")
        return cls(**values)  # type: ignore[arg-type]

    def as_dict(self) -> Dict[str, object]:
        return {
            "switching_frequency_khz": self.switching_frequency_khz,
            "dv_dt_v_per_us": self.dv_dt_v_per_us,
            "cable_length_m": self.cable_length_m,
            "shielding_quality": self.shielding_quality,
            "load_current_a": self.load_current_a,
            "pwm_modulation_type": self.pwm_modulation_type,
            "input_filter_quality": self.input_filter_quality,
            "cm_choke_effectiveness": self.cm_choke_effectiveness,
            "clock_frequency_mhz": self.clock_frequency_mhz,
            "di_dt_a_per_us": self.di_dt_a_per_us,
            "switching_device_type": self.switching_device_type,
            "dc_bus_voltage_v": self.dc_bus_voltage_v,
            "dc_bus_esl_h": self.dc_bus_esl_h,
            "dc_bus_esr_ohm": self.dc_bus_esr_ohm,
            "dead_time_us": self.dead_time_us,
            "spread_spectrum": self.spread_spectrum,
            "spread_spectrum_jitter": self.spread_spectrum_jitter,
            "rectifier_type": self.rectifier_type,
            "dc_link_choke_h": self.dc_link_choke_h,
            "y_capacitance_f": self.y_capacitance_f,
        }

    @property
    def pwm_cm_penalty_db(self) -> float:
        """Modelled CM-emission offset of this strategy relative to SPWM, in dB."""
        return PWM_CM_PENALTY_DB[self.pwm_modulation_type]

    @property
    def device_edge_multiplier(self) -> float:
        """SiC/GaN speed-up applied to nominal dv/dt and di/dt. Not an ML feature."""
        return SWITCHING_DEVICE_EDGE_MULTIPLIER[self.switching_device_type]

    @property
    def effective_dv_dt_v_per_us(self) -> float:
        return float(self.dv_dt_v_per_us) * self.device_edge_multiplier

    @property
    def effective_di_dt_a_per_us(self) -> float:
        return float(self.di_dt_a_per_us) * self.device_edge_multiplier

    def deterministic_seed(self) -> int:
        """Stable seed so identical parameters always give an identical report."""
        # Neutral advanced settings are omitted so an untouched design keeps
        # the seed, and therefore the noise floor, of the previous simulator.
        neutral = {
            "dc_bus_voltage_v": V_DC_LINK,
            "dc_bus_esl_h": 0.0,
            "dc_bus_esr_ohm": 0.0,
            "dead_time_us": 0.0,
            "spread_spectrum": False,
            "spread_spectrum_jitter": SPREAD_SPECTRUM_JITTER,
            "rectifier_type": "DIODE_6PULSE",
            "dc_link_choke_h": 0.0,
            "y_capacitance_f": 0.0,
        }
        payload = "|".join(
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(self.as_dict().items())
            if k not in neutral or v != neutral[k]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class SignalTrace:
    """Down-sampled time-domain trace for the signal explorer and the PDF."""

    key: str
    label: str
    unit: str
    timescale: str
    description: str
    time_s: np.ndarray
    values: np.ndarray


@dataclass
class SimulationResult:
    """Time-domain output of one simulation run.

    ``measured_segments`` has shape ``(N_SEGMENTS, N_SAMPLES)``; each row is one
    receiver dwell window at a different phase of the motor fundamental.
    ``time_s`` and ``cm_voltage_v`` describe the first HF segment. ``explorer``
    holds the five down-sampled signals the dashboard plots. ``pq_*`` arrays are
    the longer, slower records used for THD / harmonic analysis.
    """

    time_s: np.ndarray
    cm_voltage_v: np.ndarray            # inverter common-mode voltage (V), segment 0
    measured_segments: np.ndarray       # disturbance voltage at the LISN (V)
    sample_rate_hz: float
    parameters: DeviceParameters
    diagnostics: Dict[str, float] = field(default_factory=dict)
    explorer: Tuple[SignalTrace, ...] = ()
    pq_time_s: Optional[np.ndarray] = None
    pq_sample_rate_hz: float = PQ_SAMPLE_RATE_HZ
    pq_motor_current_a: Optional[np.ndarray] = None
    pq_input_current_a: Optional[np.ndarray] = None
    pq_dc_link_v: Optional[np.ndarray] = None

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
    phase_rng = rng
    # A dedicated stream so enabling dither does not reshuffle the noise floor.
    # RANDOM_SPWM already jitters on the main stream; leave that path alone.
    if params.spread_spectrum and jitter < float(params.spread_spectrum_jitter):
        jitter = float(np.clip(params.spread_spectrum_jitter, 0.0, 0.40))
        phase_rng = np.random.default_rng(params.deterministic_seed() ^ 0x5A17)
    carrier = _triangle_from_phase(
        _carrier_phase(t, dt, carrier_hz, jitter, phase_rng)
    )
    ref = _modulation_references(t, params.pwm_modulation_type)
    states = np.where(ref > carrier[:, None, :], 1.0, -1.0)
    return _apply_dead_time(states, params.dead_time_us * 1e-6, dt, ref)


def _apply_dead_time(
    states: np.ndarray,
    dead_time_s: float,
    dt: float,
    reference: np.ndarray,
) -> np.ndarray:
    """Blank one edge for ``dead_time_s``, in the direction of the phase current.

    Mohan, Undeland and Robbins, Power Electronics (3rd ed.): during blanking
    the leg voltage follows the device that is still carrying current, so the
    volt-second error reverses with the current. That square-wave error is a
    secondary source of low-order distortion (5th, 7th). The reference sine is
    the stand-in for current polarity. Zero dead time returns the states
    unchanged.
    """
    n_delay = int(round(dead_time_s / dt)) if dead_time_s > 0.0 else 0
    if n_delay <= 0:
        return states
    delayed = np.empty_like(states)
    delayed[..., :n_delay] = states[..., :1]
    delayed[..., n_delay:] = states[..., :-n_delay]
    positive = reference > 0.0
    blank_rise = (states > 0.0) & (delayed < 0.0) & positive
    blank_fall = (states < 0.0) & (delayed > 0.0) & ~positive
    return np.where(blank_rise | blank_fall, delayed, states)


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


def _leg_voltages(
    states: np.ndarray, params: DeviceParameters, dt: float
) -> np.ndarray:
    """Trapezoidal pole voltages, shape ``(n_segments, 3, n_samples)`` in volts."""
    rise_time_s = _dc_voltage(params) / (params.effective_dv_dt_v_per_us * 1e6)
    n_rise = max(1, int(round(rise_time_s / dt)))

    transitions = np.zeros_like(states)
    transitions[..., 1:] = np.diff(states, axis=-1) / 2.0
    spread = _causal_box_filter(transitions, n_rise)
    return (np.cumsum(spread, axis=-1) - 0.5) * _dc_voltage(params)


def _dc_voltage(params: DeviceParameters) -> float:
    """Instantaneous DC-bus setpoint. Default 565 V is the historical constant."""
    return float(params.dc_bus_voltage_v)


def _common_mode_voltage(
    states: np.ndarray, params: DeviceParameters, dt: float
) -> np.ndarray:
    """Trapezoidal leg voltages averaged into the common-mode voltage."""
    return _leg_voltages(states, params, dt).mean(axis=1)


def _rl_current(voltage_v: np.ndarray, dt: float, inductance_h: float,
                resistance_ohm: float) -> np.ndarray:
    """Causal RL current for ``v = L di/dt + R i``, last axis is time.

    Exponential Euler is stable at both the 200 MS/s EMC rate and the 200 kS/s
    power-quality rate. Current ripple amplitude falls as 1/(L · f_sw) for a
    PWM voltage, which is the relationship the motor-current explorer must show.
    """
    decay = float(np.exp(-resistance_ohm * dt / max(inductance_h, 1e-9)))
    gain = (1.0 - decay) / max(resistance_ohm, 1e-9)
    return signal.lfilter([gain], [1.0, -decay], voltage_v, axis=-1)


def _dc_link_voltage(
    t: np.ndarray, params: DeviceParameters, states: np.ndarray
) -> np.ndarray:
    """DC-link voltage: 300 Hz rectifier ripple plus a 2·f_sw switching residual.

    Ripple amplitude grows with load current (more charge pulled from C_dc per
    PWM cycle) and the switching residual's frequency tracks the carrier, so
    raising f_sw both raises the residual's frequency and shrinks its amplitude.
    """
    i_dc = params.load_current_a * RECTIFIER_ID_SCALE
    omega_300 = 2.0 * np.pi * 6.0 * MAINS_HZ
    ripple_300 = (i_dc / (DC_LINK_CAP_F * omega_300)) * np.sin(omega_300 * t)

    f_sw = params.switching_frequency_khz * 1e3
    # Three-phase inverter DC current has a strong component at twice the carrier.
    residual_amp = i_dc / (DC_LINK_CAP_F * 2.0 * np.pi * max(f_sw, 1e3) * 2.0)
    switching = states.mean(axis=1)  # (n_seg, n) in [-1, 1]
    # A series DC choke raises the impedance in front of C_dc, so both the
    # 300 Hz ripple and the switching residual shrink. Mohan treats the DC
    # choke as a current smoother; it is not the common-mode choke.
    # |V_c / V_source| = 1 / sqrt(1 + (ω^2 L C)^2). L = 0 leaves the gain at 1.
    inductance = max(float(params.dc_link_choke_h), 0.0)

    def _choke_gain(omega: float) -> float:
        if inductance < 1e-8:
            return 1.0
        return float(1.0 / np.sqrt(1.0 + (omega ** 2 * inductance * DC_LINK_CAP_F) ** 2))

    ripple_300 = ripple_300 * _choke_gain(omega_300)
    residual_amp = residual_amp * _choke_gain(2.0 * np.pi * 2.0 * max(f_sw, 1e3))
    return _dc_voltage(params) + ripple_300 + residual_amp * switching


def _input_current(
    t: np.ndarray, params: DeviceParameters, dt: float, rng: np.random.Generator
) -> np.ndarray:
    """Mains-side phase current of a 6-pulse diode rectifier plus a line reactor.

    A 6-pulse bridge produces the textbook odd-non-triplen series (5th, 7th,
    11th, 13th, ...). Load current scales the amplitude; a small extra distortion
    term grows with load so THD worsens at heavy duty, matching the typical VFD
    observation that a larger DC-link ripple at high current feeds back into the
    AC side. The line reactor (``input_filter_quality``) is a first-order lag
    that preferentially attenuates those harmonics.
    """
    i_dc = params.load_current_a * RECTIFIER_ID_SCALE
    theta = 2.0 * np.pi * MAINS_HZ * t
    if params.rectifier_type == "ACTIVE_FRONT_END":
        # PWM rectifier. Mohan: an active front end draws a near-sinusoid,
        # with residual harmonics of a few percent, against the ~30% THD of
        # an uncontrolled six-pulse bridge (5th and 7th dominant).
        current = i_dc * (
            np.cos(theta) + 0.03 * np.cos(5.0 * theta) + 0.02 * np.cos(7.0 * theta)
        )
    else:
        # Classical 6-pulse Fourier series, truncated at the 25th.
        current = np.zeros_like(t, dtype=float)
        two_root3_over_pi = 2.0 * np.sqrt(3.0) / np.pi
        load_distortion = 1.0 + 0.35 * (params.load_current_a / LOAD_CURRENT_REF_A)
        for harmonic in (1, 5, 7, 11, 13, 17, 19, 23, 25):
            sign = 1.0 if harmonic % 4 == 1 else -1.0  # +1, -5, +7, -11, ...
            weight = two_root3_over_pi * sign / harmonic
            if harmonic > 1:
                weight *= load_distortion
            current = current + weight * np.cos(harmonic * theta)
        current = current * i_dc

    # Switching-frequency hash on the DC bus, coupled back through C_dc.
    f_sw = params.switching_frequency_khz * 1e3
    hf = 0.04 * i_dc * np.sin(2.0 * np.pi * 2.0 * f_sw * t)
    hf *= (1.0 + 0.5 * rng.standard_normal(t.shape).clip(-1.0, 1.0) * 0.05)
    current = current + hf

    q = float(np.clip(params.input_filter_quality, 0.0, 1.0))
    l_f = q * INPUT_FILTER_L_MAX_H
    if l_f > 1e-6:
        r_eq = 0.25
        current = _rl_current(current * r_eq, dt, l_f, r_eq)
    return current


def _downsample_trace(time_s: np.ndarray, values: np.ndarray,
                      n_points: int = EXPLORER_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    if values.size <= n_points:
        return time_s, values
    index = np.linspace(0, values.size - 1, n_points).astype(int)
    return time_s[index], values[index]


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


def _apply_cm_choke(
    i_cm: np.ndarray, params: DeviceParameters, sample_rate_hz: float
) -> Tuple[np.ndarray, float]:
    """First-order CM-choke low-pass. Returns (current, cutoff Hz).

    See CM_CHOKE_L_MAX_H. A zero effectiveness is an identity.
    """
    q = float(np.clip(params.cm_choke_effectiveness, 0.0, 1.0))
    if q < 1e-3:
        return i_cm, float("inf")
    inductance = q * CM_CHOKE_L_MAX_H
    cutoff_hz = CM_CHOKE_Z_OHM / (2.0 * np.pi * inductance)
    wn = float(np.clip(cutoff_hz / (0.5 * sample_rate_hz), 1e-6, 0.98))
    b, a = signal.butter(1, wn, btype="low")
    return signal.lfilter(b, a, i_cm, axis=-1), float(cutoff_hz)


def _bus_ring_voltage(
    states: np.ndarray, params: DeviceParameters, dt: float
) -> np.ndarray:
    """Lightly damped bus-bar ring, volts, shape ``(n_segments, n_samples)``.

    f = 1 / (2 π √(L C_film)). A current step into that tank rings at about
    I · √(L/C) (Erickson and Maksimovic, Fundamentals of Power Electronics).
    ESR sets the damping, ζ = (R/2) √(C/L). L = 0 is an exact zero.
    """
    single = states.ndim == 2
    if single:
        states = states[None, ...]
    n_segments, _legs, n_samples = states.shape
    inductance = float(params.dc_bus_esl_h)
    if inductance < 1e-12:
        zeros = np.zeros((n_segments, n_samples))
        return zeros[0] if single else zeros

    omega = 1.0 / np.sqrt(inductance * BUS_FILM_C_F)
    resistance = float(params.dc_bus_esr_ohm)
    if resistance > 0.0:
        zeta = (resistance / 2.0) * np.sqrt(BUS_FILM_C_F / inductance)
    else:
        zeta = 0.08
    zeta = float(np.clip(zeta, 0.02, 0.85))
    edges = np.zeros((n_segments, n_samples))
    edges[:, 1:] = np.abs(np.diff(states.mean(axis=1), axis=-1))
    b, a = signal.bilinear(
        [omega ** 2], [1.0, 2.0 * zeta * omega, omega ** 2], fs=1.0 / dt
    )
    impulse = np.zeros(n_samples)
    impulse[0] = 1.0
    peak = float(np.max(np.abs(signal.lfilter(b, a, impulse)))) + 1e-30
    ring = signal.lfilter(b, a, edges, axis=-1) / peak
    amplitude = float(params.load_current_a) * np.sqrt(inductance / BUS_FILM_C_F)
    out = amplitude * ring
    return out[0] if single else out


def _apply_y_capacitor(
    i_cm: np.ndarray, params: DeviceParameters, sample_rate_hz: float
) -> np.ndarray:
    """Shunt Y-capacitor, separate from the series common-mode choke.

    The current that reaches the 50 ohm LISN is the divider
    Z_y / (Z_y + R_lisn), with Z_y = R_esr + s L_esl + 1/(s C).
    C = 0 is an identity. L_esl and R_esr are fixed parasitics (see
    Y_CAP_ESL_H): the ideal capacitor would keep falling forever, but a real
    Y-cap bottoms out at its self-resonance and then recovers as it becomes
    inductive. A choke blocks common-mode current; a Y-cap shunts it.
    """
    capacitance = float(params.y_capacitance_f)
    if capacitance < 1e-13:
        return i_cm
    inductance = Y_CAP_ESL_H
    resistance = Y_CAP_ESR_OHM
    numerator = [inductance * capacitance, resistance * capacitance, 1.0]
    denominator = [
        inductance * capacitance,
        (resistance + LISN_IMPEDANCE_OHM) * capacitance,
        1.0,
    ]
    b, a = signal.bilinear(numerator, denominator, fs=sample_rate_hz)
    return signal.lfilter(b, a, i_cm, axis=-1)


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
    legs = _leg_voltages(states, params, dt)
    ring = _bus_ring_voltage(states, params, dt)
    if float(params.dc_bus_esl_h) >= 1e-12:
        legs = legs * (1.0 + ring[:, None, :] / _dc_voltage(params))
    v_cm = legs.mean(axis=1)

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
    i_cm, choke_cutoff_hz = _apply_cm_choke(i_cm, params, sample_rate_hz)
    i_cm = _apply_y_capacitor(i_cm, params, sample_rate_hz)

    # Disturbance voltage developed across the measuring impedance, plus a
    # randomised instrument / environment noise floor (step 6).
    noise_floor_dbuv = float(rng.uniform(*NOISE_FLOOR_DBUV_RANGE))
    v_meas = i_cm * LISN_IMPEDANCE_OHM * COUPLING_CALIBRATION
    v_meas = v_meas + _noise_floor((n_segments, total), noise_floor_dbuv, rng)

    measured = np.ascontiguousarray(v_meas[:, N_WARMUP_SAMPLES:])
    time_s = np.arange(n_samples) * dt
    cm_keep = v_cm[0, N_WARMUP_SAMPLES:]
    legs_keep = legs[0, :, N_WARMUP_SAMPLES:]
    motor_voltage = legs_keep[0] - legs_keep[1]

    # Power-quality record: long enough to resolve 50 Hz THD, cheap enough to
    # sit beside the EMC capture on every request.
    pq_dt = 1.0 / PQ_SAMPLE_RATE_HZ
    pq_t = np.arange(PQ_N_SAMPLES) * pq_dt
    pq_rng = np.random.default_rng((params.deterministic_seed() if seed is None else seed) + 17)
    pq_states = _switching_states(params, pq_t[None, :], pq_dt, pq_rng)[0]
    pq_legs = _leg_voltages(pq_states[None, ...], params, pq_dt)[0]
    pq_motor_current = _rl_current(pq_legs[0], pq_dt, MOTOR_L_H, MOTOR_R_OHM)
    pq_dc_link = _dc_link_voltage(pq_t, params, pq_states[None, ...])[0]
    pq_dc_link = pq_dc_link + _bus_ring_voltage(pq_states, params, pq_dt)
    pq_input = _input_current(pq_t, params, pq_dt, pq_rng)

    def _trace(key: str, label: str, unit: str, timescale: str,
               description: str, t_arr: np.ndarray, y: np.ndarray) -> SignalTrace:
        ts, ys = _downsample_trace(t_arr, y)
        return SignalTrace(key, label, unit, timescale, description, ts, ys)

    explorer = (
        _trace(
            "dc_link", "DC-link voltage", "V",
            "switching" if float(params.dc_bus_esl_h) >= 1e-12 else "fundamental",
            (
                "Switching-timescale bus voltage. Parasitic inductance rings after "
                "each transition; that ring is absent when ESL is zero."
                if float(params.dc_bus_esl_h) >= 1e-12 else
                "300 Hz rectifier ripple plus a 2×carrier residual. Ripple grows with "
                "load current; the residual frequency tracks the switching frequency."
            ),
            time_s if float(params.dc_bus_esl_h) >= 1e-12 else pq_t,
            (
                _dc_voltage(params) + ring[0, N_WARMUP_SAMPLES:]
                if float(params.dc_bus_esl_h) >= 1e-12 else pq_dc_link
            ),
        ),
        _trace(
            "motor_voltage", "Motor line-to-line voltage", "V", "switching",
            "PWM pole-to-pole voltage with trapezoidal edges set by dv/dt.",
            time_s, motor_voltage,
        ),
        _trace(
            "motor_current", "Motor phase current", "A", "fundamental",
            "Stator current after RL filtering. Ripple amplitude falls as 1/(L·f_sw).",
            pq_t, pq_motor_current,
        ),
        _trace(
            "common_mode", "Common-mode voltage", "V", "switching",
            "v_cm = (v_a + v_b + v_c)/3, the primary driver of conducted EMC.",
            time_s, cm_keep,
        ),
        _trace(
            "input_current", "Input current", "A", "fundamental",
            "6-pulse rectifier phase current. 5th and 7th harmonics dominate; "
            "a line reactor (input filter) attenuates them."
            if params.rectifier_type == "DIODE_6PULSE" else
            "Active-front-end phase current: near-sinusoidal, with only a small "
            "5th and 7th residual. Much lower distortion than a 6-pulse bridge.",
            pq_t, pq_input,
        ),
    )

    return SimulationResult(
        time_s=time_s,
        cm_voltage_v=cm_keep,
        measured_segments=measured,
        sample_rate_hz=sample_rate_hz,
        parameters=params,
        diagnostics={
            "n_segments": float(n_segments),
            "rise_time_ns": _dc_voltage(params) / (params.effective_dv_dt_v_per_us * 1e6) * 1e9,
            "envelope_corner_hz": params.effective_dv_dt_v_per_us * 1e6 / (np.pi * _dc_voltage(params)),
            "bus_ring_pp_v": float(np.ptp(ring)),
            "cable_resonance_hz": f_res,
            "cable_capacitance_pf": c_par_f * 1e12,
            "shield_insertion_loss_db": insertion_loss_db,
            "shield_cutoff_hz": shield_cutoff_hz,
            "cm_choke_cutoff_hz": choke_cutoff_hz,
            "effective_dv_dt_v_per_us": params.effective_dv_dt_v_per_us,
            "noise_floor_dbuv": noise_floor_dbuv,
            "load_gain": load_gain,
            "motor_l_mh": MOTOR_L_H * 1e3,
            "input_filter_quality": float(params.input_filter_quality),
        },
        explorer=explorer,
        pq_time_s=pq_t,
        pq_sample_rate_hz=PQ_SAMPLE_RATE_HZ,
        pq_motor_current_a=pq_motor_current,
        pq_input_current_a=pq_input,
        pq_dc_link_v=pq_dc_link,
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
        input_filter_quality=_uniform("input_filter_quality"),
        cm_choke_effectiveness=_uniform("cm_choke_effectiveness"),
        clock_frequency_mhz=_uniform("clock_frequency_mhz"),
        di_dt_a_per_us=_log_uniform("di_dt_a_per_us"),
        switching_device_type=str(rng.choice(list(SWITCHING_DEVICE_TYPES))),
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
