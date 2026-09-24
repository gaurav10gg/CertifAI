"""
Radiated-emissions sub-model, 30 MHz to 1 GHz.

This is not an extension of the conducted common-mode model. Conducted emissions
are a current on a wire into a LISN, in dBuV. Radiated emissions are a field in
the air, in dBuV/m, at a stated distance.

Two mechanisms, both standard digital-EMC estimates (Henry Ott, Electromagnetic
Compatibility Engineering, Wiley, 2009, chapters on digital spectra and on the
small loop):

  * A clock is a trapezoidal pulse train. Its spectrum is discrete lines at
    every integer harmonic. The envelope falls as |sinc(f * t_rise)|, the
    Fourier transform of a finite rise time (Ott's sinc envelope; the same
    shape is often written sinc-squared when the quantity is power).
  * A switching current in a loop radiates a magnetic field that scales with
    di/dt and with loop area. Cable length times a fixed 5 cm spacing stands
    in for that area. There is no PCB-loop input.

Cable resonance reuses the transmission-line factor sin^2(pi L / lambda): the
field is multiplied by |sin(pi L f / c)|, which peaks when the cable is an odd
number of half-wavelengths.

The limit line is a synthetic step anchored on the publicly described CISPR 11
/ EN 55011 Group 1 Class A radiated limits at 10 m (quasi-peak): 40 dBuV/m
from 30 to 230 MHz and 47 dBuV/m from 230 MHz to 1 GHz. It is not the
copyrighted EN 12016 table, and it is not a certification limit.

The score is a separate exploratory indicator. It must not be mixed into the
conducted risk score.
"""

from __future__ import annotations

from typing import Dict, Final, List, Sequence, Tuple

import numpy as np

from simulate import DeviceParameters

C_LIGHT_M_S: Final[float] = 2.99792458e8
F_LOW_HZ: Final[float] = 30e6
F_HIGH_HZ: Final[float] = 1e9
# Fixed harness spacing used only as a stand-in for loop width.
LOOP_SPACING_M: Final[float] = 0.05
# Unscaled digital rise time before the SiC/GaN multiplier. A few nanoseconds
# is the usual logic-clock edge in Ott's examples.
CLOCK_RISE_TIME_S: Final[float] = 2.0e-9
# Chosen so a mid-range design sits near the Class A line. Same role as
# COUPLING_CALIBRATION in the conducted model: an assumption, not a measurement.
RADIATED_LEVEL_OFFSET_DB: Final[float] = 33.0
FLOOR_DB: Final[float] = 15.0

RADIATED_BANDS: Final[Tuple[Tuple[str, str, float, float], ...]] = (
    ("rad_low", "30 MHz - 230 MHz", 30e6, 230e6),
    ("rad_high", "230 MHz - 1 GHz", 230e6, 1e9),
)

# CISPR 11 Group 1 Class A, 10 m, quasi-peak, as publicly summarised for
# EN 55011. The step at 230 MHz is the standard's structure. We do not reprint
# a copyrighted table and we do not claim EN 12016.
RADIATED_LIMIT_DBUVM: Final[Tuple[Tuple[float, float, float], ...]] = (
    (30e6, 230e6, 40.0),
    (230e6, 1e9, 47.0),
)

RADIATED_LIMIT_DESCRIPTION: Final[str] = (
    "Synthetic radiated limit anchored on the publicly described CISPR 11 "
    "Group 1 Class A quasi-peak limits at 10 m: 40 dBuV/m from 30 to 230 MHz "
    "and 47 dBuV/m from 230 MHz to 1 GHz. This is an assumption of this tool, "
    "not the normative EN 12016 table, and the unit is field strength, not "
    "the conducted dBuV."
)

# risk_sign +1 means a larger value can only raise radiated risk.
RADIATED_DESIGN_FEATURES: Final[Tuple[Tuple[str, int, str], ...]] = (
    ("clock_frequency_mhz", +1, "A higher clock packs more harmonics into 30 MHz-1 GHz."),
    ("di_dt_a_per_us", +1, "Loop H-field scales with di/dt."),
    ("cable_length_m", +1, "Longer cable is a larger current loop and a longer antenna."),
    ("device_edge_multiplier", +1, "SiC and GaN shorten the clock edge and raise di/dt."),
    ("rad_low_peak_dbuvm", +1, "Highest level in 30-230 MHz."),
    ("rad_low_resonance", +1, "Peak |sin(pi L/lambda)| in 30-230 MHz."),
    ("rad_high_peak_dbuvm", +1, "Highest level in 230 MHz-1 GHz."),
    ("rad_high_resonance", +1, "Peak |sin(pi L/lambda)| in 230 MHz-1 GHz."),
)


def radiated_limit_dbuvm(frequency_hz: np.ndarray) -> np.ndarray:
    freq = np.asarray(frequency_hz, dtype=float)
    limit = np.full(freq.shape, 47.0)
    limit[freq < 230e6] = 40.0
    return limit


def _sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x, dtype=float)
    big = np.abs(x) > 1e-8
    out[big] = np.abs(np.sin(x[big]) / x[big])
    return out


def clock_harmonic_frequencies_hz(clock_hz: float) -> np.ndarray:
    n_max = int(np.floor(F_HIGH_HZ / clock_hz))
    harmonics = clock_hz * np.arange(1, n_max + 1, dtype=float)
    return harmonics[harmonics >= F_LOW_HZ]


def radiated_spectrum(params: DeviceParameters) -> Dict[str, np.ndarray]:
    """Discrete clock lines plus the loop and cable-resonance weighting."""
    clock_hz = float(params.clock_frequency_mhz) * 1e6
    freqs = clock_harmonic_frequencies_hz(clock_hz)
    rise = CLOCK_RISE_TIME_S / max(params.device_edge_multiplier, 1e-6)
    area = max(float(params.cable_length_m), 0.05) * LOOP_SPACING_M
    di_dt = params.effective_di_dt_a_per_us * 1e6  # A/s
    sinc = _sinc(np.pi * freqs * rise)
    resonance = np.abs(np.sin(np.pi * float(params.cable_length_m) * freqs / C_LIGHT_M_S))
    # Field proxy. Division by f is the small-loop far-field roll-off (Ott).
    amplitude = di_dt * area * sinc * (0.05 + resonance) / freqs
    level = 20.0 * np.log10(np.maximum(amplitude, 1e-30)) + RADIATED_LEVEL_OFFSET_DB
    return {
        "frequency_hz": freqs,
        "emission_dbuvm": level,
        "limit_dbuvm": radiated_limit_dbuvm(freqs),
        "resonance": resonance,
    }


def radiated_features(params: DeviceParameters) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    spec = radiated_spectrum(params)
    freqs = spec["frequency_hz"]
    level = spec["emission_dbuvm"]
    resonance = spec["resonance"]
    bands: List[Dict[str, float]] = []
    values: List[float] = [
        float(params.clock_frequency_mhz),
        float(params.di_dt_a_per_us),
        float(params.cable_length_m),
        float(params.device_edge_multiplier),
    ]
    for key, label, f_lo, f_hi in RADIATED_BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(mask):
            peak = FLOOR_DB
            res = 0.0
            peak_hz = float(f_lo)
            limit_at_peak = float(radiated_limit_dbuvm(np.array([f_lo]))[0])
            margin = limit_at_peak - peak
            n_lines = 0
        else:
            peak_index = int(np.argmax(level[mask]))
            peak = float(level[mask][peak_index])
            res = float(np.max(resonance[mask]))
            peak_hz = float(freqs[mask][peak_index])
            limit_at_peak = float(spec["limit_dbuvm"][mask][peak_index])
            margin = limit_at_peak - peak
            n_lines = int(np.count_nonzero(mask))
        bands.append({
            "key": key,
            "label": label,
            "f_low_hz": float(f_lo),
            "f_high_hz": float(f_hi),
            "peak_dbuvm": peak,
            "peak_frequency_hz": peak_hz,
            "limit_dbuvm": limit_at_peak,
            "resonance": res,
            "margin_db": margin,
            "n_lines": n_lines,
        })
        values.extend([peak, res])
    vector = np.asarray(values, dtype=float)
    assert vector.size == len(RADIATED_DESIGN_FEATURES)
    return vector, bands


def radiated_margins(params: DeviceParameters) -> np.ndarray:
    _vector, bands = radiated_features(params)
    return np.asarray([b["margin_db"] for b in bands], dtype=float)
