"""Print clock harmonics and compare Si vs GaN edge speed. No model required."""

from simulate import DeviceParameters, simulate_device
from radiated import radiated_spectrum


def edge(kind: str) -> None:
    params = DeviceParameters(
        switching_frequency_khz=8,
        dv_dt_v_per_us=3000,
        cable_length_m=30,
        shielding_quality=0.7,
        load_current_a=45,
        switching_device_type=kind,
    )
    sim = simulate_device(params, n_segments=1)
    print(
        kind,
        "effective_dv_dt", round(params.effective_dv_dt_v_per_us, 1),
        "rise_ns", round(sim.diagnostics["rise_time_ns"], 2),
    )


def harmonics() -> None:
    params = DeviceParameters(
        switching_frequency_khz=8,
        dv_dt_v_per_us=3000,
        cable_length_m=30,
        shielding_quality=0.7,
        load_current_a=45,
        clock_frequency_mhz=48,
    )
    spec = radiated_spectrum(params)
    freqs = spec["frequency_hz"]
    level = spec["emission_dbuvm"]
    limit = spec["limit_dbuvm"]
    print("clock lines (expect 48, 96, 144, ... MHz)")
    for index in range(6):
        print(
            f"  n={index + 1:d}  {freqs[index] / 1e6:7.2f} MHz"
            f"  {level[index]:6.1f} dBuV/m  limit {limit[index]:.0f}"
        )
    print("spacing_MHz", round((freqs[1] - freqs[0]) / 1e6, 3))


if __name__ == "__main__":
    edge("SI_IGBT")
    edge("GAN")
    harmonics()
