"""Read a schematic PDF and pre-fill the design parameters it can support.

A schematic gives you the hardware: device family, gate resistors, common-mode
chokes, Y-capacitors, DC-link capacitance, ferrite beads, MOVs, and grounding
notes. It does not give you the switching frequency (firmware), the cable
(installation), the shielding (installation), or the load current (rating).
So this module never produces a complete ``DeviceParameters``. It produces

  * a bill of materials with every refdes + value pair it could read,
  * a list of EMC-relevant findings, each pointing at the refdes it came from,
  * a partial parameter dict with a confidence and source for each field,
  * the list of fields the user must still supply.

Everything is rule-based on the PDF text layer (pdfplumber). No model is
trained here. A scanned PDF with no text layer is reported as such rather
than guessed at.

Value grammar
-------------
CAD exports print values in a handful of forms, all handled by ``parse_value``:

    100nF  10n  1n  4.7uF  0.01UF  120uF  47k5  47.5K  221k  33.0R  0.00R
    6.8uH  750uH  4X500UH  47uH  2200R±25%@100MHz  600V  1A-1000V-75ns

``47k5`` is the IEC letter-as-decimal-point form (47.5 kΩ). ``4X500UH`` is a
four-winding choke, read as 4 × 500 µH.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from simulate import CM_CHOKE_L_MAX_H, PARAMETER_RANGE_BY_KEY

# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------
_SI: Dict[str, float] = {
    "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3,
    "": 1.0, "k": 1e3, "K": 1e3, "M": 1e6,
}

# 47k5 / 4R7 / 1n0 style: digits, a unit letter acting as the decimal point,
# then more digits.
_LETTER_POINT = re.compile(r"^(\d+)([pnuµmkKMR])(\d+)$")
# 100nF / 4.7uF / 33.0R / 221k / 0.01UF / 6.8uH
_PLAIN = re.compile(r"^(\d+(?:\.\d+)?)\s*([pnuµmkKM]?)\s*([FHRΩ]|OHM)?$", re.IGNORECASE)
# 4X500UH multi-winding
_MULTI = re.compile(r"^(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*([pnuµmkKM]?)\s*([FHR])?$", re.IGNORECASE)


def parse_value(text: str) -> Optional[Tuple[float, str, int]]:
    """Return (value_si, kind, windings) or None. kind is 'C', 'L', 'R' or ''."""
    t = text.strip().replace("Ω", "R").replace("ohm", "R").replace("OHM", "R")
    t = t.split("±")[0].split("@")[0].strip()

    m = _MULTI.match(t)
    if m:
        n, num, prefix, unit = m.groups()
        kind = {"F": "C", "H": "L", "R": "R"}.get((unit or "").upper(), "")
        return float(num) * _SI.get(prefix.lower() if prefix not in ("K", "M") else prefix, 1.0), kind, int(n)

    m = _LETTER_POINT.match(t)
    if m:
        whole, letter, frac = m.groups()
        val = float(f"{whole}.{frac}")
        if letter == "R":
            return val, "R", 1
        mult = _SI.get(letter if letter in ("K", "M") else letter.lower(), 1.0)
        return val * mult, "", 1

    m = _PLAIN.match(t)
    if m:
        num, prefix, unit = m.groups()
        prefix_key = prefix if prefix in ("K", "M") else prefix.lower()
        # A bare "u"/"n"/"p" with no unit letter is almost always a capacitor
        # in schematic exports (10n, 1n, 100p).
        unit_u = (unit or "").upper()
        kind = {"F": "C", "H": "L", "R": "R", "OHM": "R"}.get(unit_u, "")
        if not kind and prefix_key in ("p", "n"):
            kind = "C"
        if unit_u == "" and prefix == "" and num:
            return None
        return float(num) * _SI.get(prefix_key, 1.0), kind, 1
    return None


# ---------------------------------------------------------------------------
# Bill of materials
# ---------------------------------------------------------------------------
@dataclass
class Part:
    refdes: str
    value_text: str
    value_si: Optional[float]
    kind: str            # C, L, R, Q, V (diode), U, D, RV, ...
    windings: int = 1
    page: int = 0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "refdes": self.refdes, "value": self.value_text, "value_si": self.value_si,
            "kind": self.kind, "windings": self.windings, "page": self.page,
        }


# A refdes token: R92, C88, L16, Q10, RV6, U27-1.
_REF_TOKEN = re.compile(r"^(?:RV|[RCLQVDU])\d{1,4}[A-Z]?(?:-\d)?$")
# Things that look like a component value or a device description.
_VALUE_TOKEN = re.compile(
    r"^(?:\d+(?:\.\d+)?[pnuµmkKMR]?\d*(?:[FHRΩ]|OHM|UF|UH|NF|PF)?"
    r"|\d+[xX]\d+(?:\.\d+)?[pnuµmkKM]?[FHR]?)$",
    re.IGNORECASE,
)
# Tokens that are net labels, page refs, or hardware, never a value.
_NOISE = re.compile(r"^(?:E\d{1,4}|TP\d+|J\d+|H\d+|XB\d+|\d(?:,\d)*|[A-H]|UDC/-|\+UDC|P\d+V\d*.*|GND.*|HOLE.*)$")

_DEVICE_WORDS = (
    ("GAN", "GAN"), ("SIC", "SIC_MOSFET"), ("IGBT", "SI_IGBT"),
    ("MOSFET", "SI_IGBT"), ("FET", "SI_IGBT"),
)
_DEVICE_LINE = re.compile(r"\b(GaN|SiC|IGBT|MOSFET|FET)\b\s*\d*\s*V?", re.IGNORECASE)
# Ferrite beads and MOVs print a description, not a plain value:
# "200mA-2200R±25%@100MHz", "Ferrite 4A", "175V-10kA", "1A-1000V-75ns".
_FERRITE_LINE = re.compile(
    r"(?:\d+mA-\d+R[^\s]*|Ferrite\s+\d+A|\d+V-\d+kA|\d+A-\d+V(?:-\d+ns)?|\d+/\d+V)",
    re.IGNORECASE,
)


def _read_pdf(pdf_bytes: bytes) -> Tuple[List[Tuple[int, str]], List[str]]:
    """Return (page, line) pairs plus a flat word list for title hunting."""
    import pdfplumber

    out: List[Tuple[int, str]] = []
    words: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for raw in text.splitlines():
                line = raw.strip()
                if line:
                    out.append((index, line))
            for w in page.extract_words() or []:
                tok = (w.get("text") or "").strip()
                if tok:
                    words.append(tok)
    return out, words


def _lines(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    lines, _words = _read_pdf(pdf_bytes)
    return lines


def _tokens(lines: Sequence[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Flatten the text layer into (page, token) pairs.

    CAD exports glue neighbouring labels onto one line ("L18 4X500UH HOLE
    5-PIN"), so pairing works on tokens, not lines. Device descriptions such
    as "IGBT 600V" or "SiC 600V" are kept as one token.
    """
    out: List[Tuple[int, str]] = []
    for page, line in lines:
        # Replace each device description with a single placeholder so it
        # keeps its position in the line, then split on spaces.
        descs: List[str] = []

        def _hold(m: "re.Match[str]") -> str:
            descs.append(m.group(0).strip())
            return f" \x00{len(descs) - 1}\x00 "

        marked = _DEVICE_LINE.sub(_hold, line)
        for m in _FERRITE_LINE.finditer(marked):
            pass
        marked = _FERRITE_LINE.sub(_hold, marked)
        for tok in marked.split():
            if tok.startswith("\x00") and tok.endswith("\x00"):
                out.append((page, descs[int(tok.strip("\x00"))]))
            else:
                out.append((page, tok))
    return out


def _kind_compatible(ref_letter: str, value_text: str) -> bool:
    """A capacitor is never '47k5'; a resistor is never '100nF'.

    Unit-less kilo/mega values belong to resistors. Values with F/H/R letters
    must match the refdes family. Anything else is left to the parser.
    """
    up = value_text.upper()
    parsed = parse_value(value_text)
    if parsed is None:
        return False
    _v, kind, _w = parsed
    if kind and ref_letter in ("C", "L", "R"):
        return kind == ref_letter
    if ref_letter == "C":
        return "K" not in up and "M" not in up.replace("MHZ", "")
    if ref_letter == "L":
        return "K" not in up
    return True


def extract_parts(lines: Sequence[Tuple[int, str]]) -> List[Part]:
    """Pair each refdes token with the first value-like token within reach."""
    tokens = _tokens(lines)
    parts: Dict[str, Part] = {}
    WINDOW = 6

    def commit(ref: str, val: str, page: int) -> None:
        parsed = parse_value(val)
        kind = "RV" if ref.startswith("RV") else ref[0]
        value_si = parsed[0] if parsed else None
        windings = parsed[2] if parsed else 1
        if parsed and parsed[1] in ("C", "L", "R") and kind in ("C", "L", "R"):
            kind = parsed[1]
        if ref not in parts or (parts[ref].value_si is None and value_si is not None):
            parts[ref] = Part(ref, val, value_si, kind, windings, page)

    def matches(ref: str, tok: str) -> bool:
        letter = "RV" if ref.startswith("RV") else ref[0]
        if _NOISE.match(tok):
            return False
        if letter in ("Q", "V"):
            return bool(_DEVICE_LINE.search(tok)) or bool(re.match(r"^\d+A-\d+V", tok))
        if letter == "RV":
            return bool(re.match(r"^\d+V-\d+kA|^\d+/\d+V", tok))
        if letter == "L" and _FERRITE_LINE.fullmatch(tok):
            return True
        if _VALUE_TOKEN.match(tok) and parse_value(tok) is not None:
            if re.fullmatch(r"\d+", tok):
                return False  # pin numbers
            return _kind_compatible(letter, tok)
        return False

    # Forward pass first (the usual "R89 10.0K" order), then a short backward
    # pass for exports that print the value above the refdes.
    for i, (page, tok) in enumerate(tokens):
        if not _REF_TOKEN.match(tok):
            continue
        ref = tok
        found = False
        for _, nxt in tokens[i + 1:i + 1 + WINDOW]:
            if _REF_TOKEN.match(nxt) and not nxt.startswith("U"):
                break
            if matches(ref, nxt):
                commit(ref, nxt, page)
                found = True
                break
        if found:
            continue
        for _, prv in reversed(tokens[max(0, i - 3):i]):
            if _REF_TOKEN.match(prv):
                break
            if matches(ref, prv):
                commit(ref, prv, page)
                break
        if ref not in parts:
            parts[ref] = Part(ref, "", None, "RV" if ref.startswith("RV") else ref[0], 1, page)

    return [p for p in parts.values() if p.value_text or p.kind in ("U", "D")]


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
_EARTH_NETS = re.compile(r"\b(GND_EARTH|PE\b|CHASSIS|EARTH|PROTECTIVE)", re.IGNORECASE)


def analyse(pdf_bytes: bytes) -> Dict[str, Any]:
    lines, words = _read_pdf(pdf_bytes)
    text_all = "\n".join(l for _, l in lines)
    if len(text_all) < 200:
        return {
            "ok": False,
            "reason": (
                "This PDF has no readable text layer. It is probably a scanned "
                "image. Export the schematic from the CAD tool as a PDF with "
                "text, or enter the parameters manually."
            ),
            "pages": 0,
            "parts": [], "findings": [], "inferred": {}, "missing": _MISSING_ALL,
        }

    parts = extract_parts(lines)
    findings: List[Dict[str, Any]] = []
    inferred: Dict[str, Dict[str, Any]] = {}

    # --- Device family ---------------------------------------------------
    device_votes: Dict[str, List[str]] = {}
    for p in parts:
        if p.kind == "Q":
            up = p.value_text.upper()
            for word, dev in _DEVICE_WORDS:
                if word in up:
                    device_votes.setdefault(dev, []).append(p.refdes)
                    break
    if device_votes:
        # Prefer the fastest family present in the power stage: if any GaN or
        # SiC switch exists, its edges set the emissions. Drop generic FET
        # votes when explicit IGBT/SiC/GaN parts are also on the sheet.
        explicit = [p for p in parts if p.kind == "Q" and re.search(r"\b(IGBT|SiC|GaN)\b", p.value_text, re.I)]
        if explicit:
            for dev, refs in list(device_votes.items()):
                keep = [r for r in refs if any(p.refdes == r and re.search(r"\b(IGBT|SiC|GaN)\b", p.value_text, re.I) for p in parts)]
                if keep:
                    device_votes[dev] = keep
                else:
                    del device_votes[dev]
        order = ["GAN", "SIC_MOSFET", "SI_IGBT"]
        dev = next(d for d in order if d in device_votes)
        refs = device_votes[dev]
        inferred["switching_device_type"] = {
            "value": dev, "confidence": 0.8 if len(refs) >= 2 else 0.6,
            "source": ", ".join(refs[:6]),
        }
        findings.append({
            "key": "device", "label": "Switching devices",
            "detail": f"{len(refs)} × {dev.replace('_', ' ')} ({', '.join(refs[:6])})",
            "refdes": refs,
            "emc_note": "IGBT edges are the slowest of the three families; SiC and GaN switch 2.5-5× faster at the same rating.",
        })

    # --- Gate resistors -> dv/dt bin -------------------------------------
    # Small resistors (5-100 Ω) placed at gate drivers. Without a netlist we
    # take resistors in that range that sit near a driver IC in the text.
    gate_r = [p for p in parts if p.kind == "R" and p.value_si is not None and 4.0 <= p.value_si <= 100.0]
    driver_present = any("HCPL" in p.value_text.upper() or "DRIVER" in p.value_text.upper() or "1ED" in p.value_text.upper()
                         for p in parts if p.kind == "U") or "HCPL" in text_all.upper()
    if gate_r and driver_present:
        vals = sorted(p.value_si for p in gate_r if p.value_si)
        median = vals[len(vals) // 2]
        # Rough mapping for a 600 V IGBT module: 10 Ω ≈ very fast, 33 Ω ≈ a
        # typical 3-5 kV/µs, 100 Ω ≈ slow. Documented as a bin, not a measurement.
        if median <= 15:
            dvdt, conf = 6000.0, 0.4
        elif median <= 40:
            dvdt, conf = 3500.0, 0.5
        elif median <= 70:
            dvdt, conf = 2200.0, 0.5
        else:
            dvdt, conf = 1200.0, 0.4
        refs = [p.refdes for p in gate_r][:8]
        inferred["dv_dt_v_per_us"] = {"value": dvdt, "confidence": conf, "source": ", ".join(refs)}
        findings.append({
            "key": "gate_resistor", "label": "Gate resistors",
            "detail": f"median {median:.0f} Ω across {len(gate_r)} parts ({', '.join(refs)})",
            "refdes": refs,
            "emc_note": f"Mapped to a nominal {dvdt:,.0f} V/µs edge. A larger gate resistor slows the edge and lowers 5-30 MHz emissions at the cost of switching loss.",
        })

    # --- Common-mode chokes -----------------------------------------------
    # Multi-winding inductors on a brake/PFC sheet (L16/L18 4x500 µH) sit on
    # the AC input, not the motor cable. The conducted score is a motor-cable
    # common-mode current, so those parts must not be written into
    # cm_choke_effectiveness or a 1 mH fill will pin the score near 100.
    _MOTOR_CM = re.compile(
        r"\b(motor|output|U/V/W|common[-\s]?mode|CM\s*choke)\b", re.IGNORECASE
    )
    multi_l = [p for p in parts if p.kind == "L" and p.windings >= 2 and p.value_si]
    motor_cm = []
    line_cm = []
    for p in multi_l:
        nearby = " ".join(
            l for pg, l in lines if pg == p.page and (_MOTOR_CM.search(l) or p.refdes in l)
        )
        if _MOTOR_CM.search(nearby) and not re.search(r"\b(PFC|AC_L|AC_N|mains)\b", nearby, re.I):
            motor_cm.append(p)
        else:
            line_cm.append(p)
    if motor_cm:
        total_h = sum(p.value_si for p in motor_cm if p.value_si)
        eff = min(1.0, total_h / CM_CHOKE_L_MAX_H)
        refs = [p.refdes for p in motor_cm]
        inferred["cm_choke_effectiveness"] = {
            "value": round(eff, 3), "confidence": 0.55, "source": ", ".join(refs),
        }
        findings.append({
            "key": "cm_choke", "label": "Motor-side common-mode choke",
            "detail": ", ".join(f"{p.refdes} {p.value_text}" for p in motor_cm),
            "refdes": refs,
            "emc_note": (
                f"About {total_h * 1e3:.2f} mH on the motor leads. That is "
                f"{eff * 2:.2f} mH on the 0–2 mH slider."
            ),
        })
    else:
        inferred["cm_choke_effectiveness"] = {
            "value": 0.0, "confidence": 0.6,
            "source": "no motor-side common-mode choke found",
        }
    if line_cm:
        findings.append({
            "key": "line_cm", "label": "Line / PFC chokes (not scored)",
            "detail": ", ".join(f"{p.refdes} {p.value_text}" for p in line_cm),
            "refdes": [p.refdes for p in line_cm],
            "emc_note": (
                "These sit on the mains or PFC input. They do not change the "
                "motor-cable conducted score, so they are not written into the "
                "common-mode choke slider."
            ),
        })

    # --- Y-capacitors ---------------------------------------------------------
    # Capacitors mentioned in the same note as an earth net, or 1-100 nF class
    # capacitors when the sheet names an earth net at all.
    earth_named = bool(_EARTH_NETS.search(text_all))
    by_ref = {p.refdes: p for p in parts}
    y_caps: List[Part] = []
    # Notes wrap across lines, so look at each earth mention plus its neighbours.
    for i, (page, line) in enumerate(lines):
        if not _EARTH_NETS.search(line):
            continue
        window = " ".join(l for _, l in lines[max(0, i - 1):i + 3])
        for ref in re.findall(r"\bC\d{1,4}\b", window):
            match = by_ref.get(ref)
            if match and match.value_si and match.value_si <= 1e-6:
                y_caps.append(match)
    y_caps = list({p.refdes: p for p in y_caps}.values())
    if y_caps:
        total_f = sum(p.value_si for p in y_caps if p.value_si)
        inferred["y_capacitance_f"] = {"value": total_f, "confidence": 0.7, "source": ", ".join(p.refdes for p in y_caps)}
        findings.append({
            "key": "y_cap", "label": "Y-capacitors to earth",
            "detail": ", ".join(f"{p.refdes} {p.value_text}" for p in y_caps),
            "refdes": [p.refdes for p in y_caps],
            "emc_note": f"{total_f * 1e9:.1f} nF shunting common-mode current to earth. Small values mainly help above a few MHz; leakage current limits how large they can be.",
        })
    else:
        inferred["y_capacitance_f"] = {
            "value": 0.0, "confidence": 0.4 if earth_named else 0.3,
            "source": "no capacitor named beside an earth net",
        }

    # --- DC link ----------------------------------------------------------------
    bulk = [p for p in parts if p.kind == "C" and p.value_si and p.value_si >= 10e-6]
    if bulk:
        total = sum(p.value_si for p in bulk if p.value_si)
        findings.append({
            "key": "dc_link", "label": "DC-link capacitance",
            "detail": f"{len(bulk)} × bulk ({', '.join(f'{p.refdes} {p.value_text}' for p in bulk[:6])})",
            "refdes": [p.refdes for p in bulk],
            "emc_note": f"About {total * 1e6:.0f} µF. Sets low-frequency ripple; the model holds its own 2200 µF for that and does not read this value.",
        })

    # --- Ferrites, MOVs, decoupling ------------------------------------------
    beads = [p for p in parts if p.kind == "L" and ("@" in p.value_text or "FERRITE" in p.value_text.upper() or "2200R" in p.value_text.upper())]
    if beads:
        findings.append({
            "key": "ferrite", "label": "Ferrite beads",
            "detail": ", ".join(f"{p.refdes} {p.value_text}" for p in beads),
            "refdes": [p.refdes for p in beads],
            "emc_note": "Beads damp 10-100 MHz resonances on signal and supply lines. They are not in the conducted model; they matter for the radiated range.",
        })
    movs = [p for p in parts if p.kind == "RV"]
    if movs:
        findings.append({
            "key": "mov", "label": "Surge protection",
            "detail": ", ".join(f"{p.refdes} {p.value_text}" for p in movs),
            "refdes": [p.refdes for p in movs],
            "emc_note": "MOVs are immunity parts (surge). They do not change emissions and are out of scope for this score.",
        })
    decoupling = [p for p in parts if p.kind == "C" and p.value_si and 1e-9 <= p.value_si <= 1e-6]
    if decoupling:
        findings.append({
            "key": "decoupling", "label": "Decoupling capacitors",
            "detail": f"{len(decoupling)} parts between 1 nF and 1 µF",
            "refdes": [p.refdes for p in decoupling][:12],
            "emc_note": "Local decoupling keeps gate-drive and logic noise off the supply. Good hygiene; not a scored parameter.",
        })

    # --- Grounding notes -----------------------------------------------------
    ground_notes = [l for _, l in lines if re.search(r"\b(GND_HS|GND_EARTH|grounding screw|one point)\b", l, re.IGNORECASE)
                    and len(l) > 25]
    if ground_notes:
        findings.append({
            "key": "grounding", "label": "Grounding notes on the drawing",
            "detail": " · ".join(ground_notes[:3]),
            "refdes": [],
            "emc_note": "Single-point heatsink-to-earth bonding and a Y-cap at the earth screw are the intended common-mode return. The model assumes this is done; a missing bond usually shows up as a failed conducted test.",
        })

    # --- Rectifier / PFC -----------------------------------------------------
    if re.search(r"UCC28180|PFC", text_all):
        findings.append({
            "key": "pfc", "label": "Active PFC front end",
            "detail": "UCC28180 controller present",
            "refdes": [p.refdes for p in parts if "UCC28180" in p.value_text.upper()],
            "emc_note": "A boost PFC stage switches its own current at the mains input. The tool's rectifier options model a 6-pulse or active front end for input THD only.",
        })
        inferred["rectifier_type"] = {"value": "ACTIVE_FRONT_END", "confidence": 0.5, "source": "UCC28180"}

    missing = [k for k in _MISSING_ALL if k not in inferred]
    parameters = {k: v["value"] for k, v in inferred.items()}

    return {
        "ok": True,
        "pages": max((pg for pg, _ in lines), default=0),
        "part_count": len(parts),
        "title": _title(lines, words),
        "parts": [p.as_dict() for p in sorted(parts, key=lambda p: (p.kind, _num(p.refdes)))],
        "findings": findings,
        "inferred": inferred,
        "parameters": parameters,
        "missing": missing,
        "note": (
            "Read from the PDF text layer with rules, not a trained model. Each "
            "inferred value names the parts it came from; check them. The fields "
            "under 'still needed' are not on a schematic: carrier frequency lives "
            "in firmware, cable and shielding are the installation, load current "
            "is the rating."
        ),
    }


_MISSING_ALL: List[str] = [
    "switching_frequency_khz", "cable_length_m", "shielding_quality", "load_current_a",
]


def _num(ref: str) -> int:
    m = re.search(r"\d+", ref)
    return int(m.group(0)) if m else 0


def _title(lines: Sequence[Tuple[int, str]], words: Optional[Sequence[str]] = None) -> str:
    for _, l in lines:
        if re.search(r"(Controller|Drive|Inverter|Converter|Board)\b", l) and 8 < len(l) < 60:
            return l
    tokens = list(words or [])
    for i, tok in enumerate(tokens):
        if tok in ("Controller", "Drive", "Inverter", "Converter") and i >= 1:
            phrase = " ".join(tokens[max(0, i - 3):i + 1])
            if 8 < len(phrase) < 60:
                return phrase
    # Title-block text is often outlined, so only the product code survives
    # as a word (BCX14). Pair it with BRAKE / DRIVE if those nets exist.
    codes = [t for t in tokens if re.fullmatch(r"[A-Z]{2,}\d{2,}", t)]
    if codes:
        kind = "brake controller" if any(t.upper() == "BRAKE" for t in tokens) else "schematic"
        return f"{codes[0]} {kind}"
    return "Uploaded schematic"
