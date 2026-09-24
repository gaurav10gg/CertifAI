"""Tests for the PDF schematic extractor (rule-based, no model involved)."""

from __future__ import annotations

from pathlib import Path

import pytest

import schematic_import as si


@pytest.mark.parametrize(
    ("text", "expected_si", "expected_kind"),
    [
        ("100nF", 100e-9, "C"),
        ("4.7uF", 4.7e-6, "C"),
        # Unit-less kilo values carry no family letter; the refdes decides.
        ("47k5", 47_500.0, ""),
        ("2200R", 2200.0, "R"),
        ("33R", 33.0, "R"),
        ("4X500UH", 500e-6, "L"),
        ("1mH", 1e-3, "L"),
    ],
)
def test_parse_value(text: str, expected_si: float, expected_kind: str) -> None:
    parsed = si.parse_value(text)
    assert parsed is not None, text
    value, kind, _windings = parsed
    assert kind == expected_kind
    assert value == pytest.approx(expected_si, rel=1e-6)


def test_parse_value_rejects_noise() -> None:
    assert si.parse_value("SHEET") is None
    assert si.parse_value("2024") is None


def test_multi_winding_choke_reports_windings() -> None:
    parsed = si.parse_value("4X500UH")
    assert parsed is not None
    assert parsed[2] == 4


def test_extract_parts_pairs_refdes_and_value() -> None:
    lines = [
        (1, "C88 1nF GND_EARTH"),
        (1, "L16 4X500UH"),
        (1, "R41 33R"),
        (2, "Q10 IGBT 600V 40A"),
    ]
    parts = si.extract_parts(lines)
    by_ref = {p.refdes: p for p in parts}
    assert by_ref["C88"].value_si == pytest.approx(1e-9)
    assert by_ref["L16"].windings == 4
    assert by_ref["R41"].value_si == pytest.approx(33.0)
    assert by_ref["Q10"].kind == "Q"


def test_kind_compatibility_blocks_resistor_value_on_capacitor() -> None:
    # "C97 1.50K" is a resistor value that drifted next to a cap refdes.
    assert not si._kind_compatible("C", "1.50K")
    assert si._kind_compatible("C", "100nF")
    assert si._kind_compatible("R", "1.50K")


BCX14 = Path.home() / "Downloads" / "50308376D02_BCX14_SHARE 1.pdf"


@pytest.mark.skipif(not BCX14.exists(), reason="sample PDF not present")
def test_analyse_bcx14_sample() -> None:
    report = si.analyse(BCX14.read_bytes())
    assert report["ok"] is True
    assert report["part_count"] > 150
    assert "BCX14" in (report["title"] or "")
    inferred = report["inferred"]
    assert inferred["switching_device_type"]["value"] == "SI_IGBT"
    # L16/L18 are PFC/line chokes. They must not fill the motor CM slider.
    assert inferred["cm_choke_effectiveness"]["value"] == pytest.approx(0.0)
    assert any(f["key"] == "line_cm" for f in report["findings"])
    assert inferred["y_capacitance_f"]["value"] == pytest.approx(1e-9, rel=0.1)
    assert set(report["missing"]) == set(si._MISSING_ALL)
    params = report["parameters"]
    assert params["switching_device_type"] == "SI_IGBT"
