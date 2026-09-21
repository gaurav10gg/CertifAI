"""
devices.py -- Preset elevator drive / controller profiles.

These are *representative* configurations covering the range of elevator power
electronics a pre-compliance user is likely to be assessing, from a cost-optimised
machine-room-less controller to a SiC regenerative drive. Parameter values are
plausible engineering figures for each archetype, not measurements from a specific
product, and no manufacturer's equipment is being characterised here.

Each profile is expressed in the same design parameters the model and the
power-quality layer consume, so selecting a preset is exactly equivalent to
typing those numbers in by hand -- the presets exist to give a user a sensible
starting point, not a different code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, List, Tuple

from simulate import DeviceParameters


@dataclass(frozen=True)
class DeviceProfile:
    """A named starting point in the design parameter space."""

    id: str
    name: str
    category: str
    summary: str
    # Short spec bullets shown on the selection card.
    highlights: Tuple[str, ...]
    # Monochrome icon key resolved by the frontend to an inline SVG.
    icon: str
    parameters: DeviceParameters

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "summary": self.summary,
            "highlights": list(self.highlights),
            "icon": self.icon,
            "parameters": self.parameters.as_dict(),
        }


DEVICE_PROFILES: Final[Tuple[DeviceProfile, ...]] = (
    DeviceProfile(
        id="standard-gearless-vfd",
        name="Standard Gearless VFD",
        category="Passenger, mid-rise",
        summary=(
            "Permanent-magnet gearless machine driven by a conventional IGBT "
            "inverter. The volume baseline for 8-20 floor passenger installations."
        ),
        highlights=(
            "8 kHz IGBT carrier",
            "30 m screened motor cable",
            "45 A rated duty",
        ),
        icon="gearless",
        parameters=DeviceParameters(
            switching_frequency_khz=8.0,
            dv_dt_v_per_us=3000.0,
            cable_length_m=30.0,
            shielding_quality=0.78,
            load_current_a=45.0,
            pwm_modulation_type="SPWM",
        ),
    ),
    DeviceProfile(
        id="high-speed-ultra-rise",
        name="High-Speed Ultra-Rise Drive",
        category="Passenger, high-rise",
        summary=(
            "High-power drive for tall-building express shafts. The very long "
            "motor cable run is the dominant EMC risk: parasitic capacitance to "
            "earth scales directly with it."
        ),
        highlights=(
            "12 kHz carrier, 6 kV/us edges",
            "85 m shaft cable run",
            "140 A rated duty",
        ),
        icon="highrise",
        parameters=DeviceParameters(
            switching_frequency_khz=12.0,
            dv_dt_v_per_us=6000.0,
            cable_length_m=85.0,
            shielding_quality=0.91,
            load_current_a=140.0,
            pwm_modulation_type="SVPWM",
        ),
    ),
    DeviceProfile(
        id="compact-mrl-controller",
        name="Compact MRL Controller",
        category="Machine-room-less",
        summary=(
            "Cost-optimised controller mounted in the shaft head. The cable run is "
            "short, but the compact enclosure and simplified screen termination "
            "leave much less filtering margin."
        ),
        highlights=(
            "16 kHz carrier for acoustic comfort",
            "12 m cable, partial screening",
            "22 A rated duty",
        ),
        icon="compact",
        parameters=DeviceParameters(
            switching_frequency_khz=16.0,
            dv_dt_v_per_us=4500.0,
            cable_length_m=12.0,
            shielding_quality=0.45,
            load_current_a=22.0,
            pwm_modulation_type="SPWM",
        ),
    ),
    DeviceProfile(
        id="freight-heavy-duty",
        name="Freight Heavy-Duty Drive",
        category="Goods and freight",
        summary=(
            "High-current goods lift drive running a low carrier frequency and "
            "discontinuous modulation to keep switching losses down at full load."
        ),
        highlights=(
            "4 kHz carrier, discontinuous PWM",
            "40 m cable, basic screening",
            "180 A rated duty",
        ),
        icon="freight",
        parameters=DeviceParameters(
            switching_frequency_khz=4.0,
            dv_dt_v_per_us=1200.0,
            cable_length_m=40.0,
            shielding_quality=0.40,
            load_current_a=180.0,
            pwm_modulation_type="DPWM",
        ),
    ),
    DeviceProfile(
        id="sic-regenerative",
        name="SiC Regenerative Drive",
        category="Next-generation",
        summary=(
            "Silicon-carbide regenerative drive. Very fast switching edges push "
            "significant energy into the 5-30 MHz band, offset here by a "
            "high-integrity screen and common-mode choke."
        ),
        highlights=(
            "16 kHz carrier, 9.5 kV/us edges",
            "28 m cable, full screen + CM choke",
            "70 A rated duty",
        ),
        icon="sic",
        parameters=DeviceParameters(
            switching_frequency_khz=16.0,
            dv_dt_v_per_us=9500.0,
            cable_length_m=28.0,
            shielding_quality=0.96,
            load_current_a=70.0,
            pwm_modulation_type="RANDOM_SPWM",
        ),
    ),
    DeviceProfile(
        id="escalator-vvvf",
        name="Escalator VVVF Unit",
        category="Escalators and walkways",
        summary=(
            "Variable-voltage variable-frequency unit for escalators and moving "
            "walks. Short, well-routed cabling and a moderate carrier give a "
            "comparatively relaxed emission profile."
        ),
        highlights=(
            "6 kHz carrier, space-vector PWM",
            "18 m cable, good screening",
            "75 A rated duty",
        ),
        icon="escalator",
        parameters=DeviceParameters(
            switching_frequency_khz=6.0,
            dv_dt_v_per_us=2200.0,
            cable_length_m=18.0,
            shielding_quality=0.70,
            load_current_a=75.0,
            pwm_modulation_type="SVPWM",
        ),
    ),
)

DEVICE_PROFILE_BY_ID: Final[Dict[str, DeviceProfile]] = {
    profile.id: profile for profile in DEVICE_PROFILES
}


def list_profiles() -> List[Dict[str, object]]:
    return [profile.as_dict() for profile in DEVICE_PROFILES]
