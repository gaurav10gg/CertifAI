"""Before/after checks for the seven advanced simulator parameters.

Run from backend/:
    .venv\\Scripts\\python.exe tools\\advanced_param_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import analyse_power_quality
from simulate import DeviceParameters, SAMPLE_RATE_HZ, simulate_device


def _base(**overrides) -> DeviceParameters:
    values = dict(
        switching_frequency_khz=8.0,
        dv_dt_v_per_us=3000.0,
        cable_length_m=30.0,
        shielding_quality=0.6,
        load_current_a=45.0,
        pwm_modulation_type="SPWM",
    )
    values.update(overrides)
    return DeviceParameters(**values)


def _maxhold_db(result) -> tuple[np.ndarray, np.ndarray]:
    segments = result.measured_segments
    window = np.hanning(segments.shape[1])
    spec = np.max(np.abs(np.fft.rfft(segments * window, axis=1)), axis=0)
    freqs = np.fft.rfftfreq(segments.shape[1], 1.0 / result.sample_rate_hz)
    db = 20.0 * np.log10(spec + 1e-18)
    return freqs, db


def _peak_near(freqs, db, center_hz, half_hz) -> float:
    mask = (freqs >= center_hz - half_hz) & (freqs <= center_hz + half_hz)
    return float(np.max(db[mask]))


def _thd(result, key: str) -> float:
    for report in analyse_power_quality(result):
        if report.key == key:
            return float(report.thd_percent)
    raise KeyError(key)


def main() -> None:
    base = simulate_device(_base())
    print("seed match", _base().deterministic_seed())

    hi = simulate_device(_base(dc_bus_voltage_v=750.0))
    f0, db0 = _maxhold_db(base)
    f1, db1 = _maxhold_db(hi)
    # First carrier harmonic inside the conducted band: 19 * 8 kHz = 152 kHz.
    p0 = _peak_near(f0, db0, 152e3, 12e3)
    p1 = _peak_near(f1, db1, 152e3, 12e3)
    print(f"1 dc bus 565->750 V   in-band peak {p0:.1f} -> {p1:.1f} dB  delta {p1-p0:+.1f}  (expect about +2.5 dB)")
    print(f"   cm rms {np.std(base.cm_voltage_v):.1f} -> {np.std(hi.cm_voltage_v):.1f} V")

    ring = simulate_device(_base(dc_bus_esl_h=200e-9, dc_bus_esr_ohm=0.02))
    print(f"2 esl 0 -> 200 nH     bus ring p-p {base.diagnostics['bus_ring_pp_v']:.2f} -> {ring.diagnostics['bus_ring_pp_v']:.2f} V  (expect a rise)")

    dead = simulate_device(_base(dead_time_us=3.0))
    t0, t1 = _thd(base, "motor_current"), _thd(dead, "motor_current")

    def _fifth(result) -> float:
        for report in analyse_power_quality(result):
            if report.key == "motor_current":
                for peak in report.peaks:
                    if peak.order == 5:
                        return float(peak.percent_of_fundamental)
                return 0.0
        return 0.0

    print(
        f"3 dead time 0 -> 3 us motor THD {t0:.2f}% -> {t1:.2f}%, "
        f"5th { _fifth(base):.2f}% -> {_fifth(dead):.2f}% of fundamental"
    )

    spread = simulate_device(_base(spread_spectrum=True))
    fs, dbs = _maxhold_db(spread)
    s0 = _peak_near(f0, db0, 152e3, 4e3)
    s1 = _peak_near(fs, dbs, 152e3, 4e3)
    band0 = (f0 >= 140e3) & (f0 <= 180e3)
    band1 = (fs >= 140e3) & (fs <= 180e3)
    # Peak-to-neighbour: how far the tallest bin stands above the median of the neighbourhood.
    stand0 = s0 - float(np.median(db0[band0]))
    stand1 = s1 - float(np.median(dbs[band1]))
    print(f"4 spread off -> on    peak {s0:.1f} -> {s1:.1f} dB, stands {stand0:.1f} -> {stand1:.1f} dB above neighbours  (expect a flatter peak)")

    afe = simulate_device(_base(rectifier_type="ACTIVE_FRONT_END"))
    i0, i1 = _thd(base, "input_current"), _thd(afe, "input_current")
    print(f"5 rectifier diode -> AFE  input THD {i0:.1f}% -> {i1:.1f}%  (expect a clear drop)")

    choke = simulate_device(_base(dc_link_choke_h=2e-3))
    r0 = float(np.ptp(base.pq_dc_link_v))
    r1 = float(np.ptp(choke.pq_dc_link_v))
    print(f"6 dc choke 0 -> 2 mH  dc-link p-p {r0:.2f} -> {r1:.2f} V  (expect a drop)")

    ycap = simulate_device(_base(y_capacitance_f=22e-9))
    fy, dby = _maxhold_db(ycap)
    y0 = _peak_near(f0, db0, 5e6, 2e6)
    y1 = _peak_near(fy, dby, 5e6, 2e6)
    print(f"7 y-cap 0 -> 22 nF    5 MHz neighbourhood {y0:.1f} -> {y1:.1f} dB  (expect a drop)")
    print("sample rate", SAMPLE_RATE_HZ)


if __name__ == "__main__":
    main()
