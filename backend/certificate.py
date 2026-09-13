"""
certificate.py -- "Virtual Pre-Compliance Assessment" PDF generation.

Deliberate naming
-----------------
The document is titled *Virtual Pre-Compliance Assessment*, never "Certificate of
Compliance". It carries no accreditation marks, no standard-conformity statement
and no signature block, because none of those would be truthful: this tool has no
accredited status and its limit line is synthetic. The disclaimer appears on the
document, not only in the app.

Styling matches the web application: strictly black, white and grey, a single
typeface family, generous leading, hairline rules instead of boxes and shading.
Pass/fail is the only place a colour appears, and even there the glyph carries the
meaning so the document survives being printed in greyscale.
"""

from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: no display, safe inside a web worker

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Design tokens -- mirror frontend/src/theme
# ---------------------------------------------------------------------------
INK = colors.HexColor("#0A0A0A")
INK_MUTED = colors.HexColor("#6B6B6B")
INK_FAINT = colors.HexColor("#9A9A9A")
RULE = colors.HexColor("#E5E5E5")
RULE_STRONG = colors.HexColor("#C9C9C9")
PAPER = colors.HexColor("#FFFFFF")
WASH = colors.HexColor("#F7F7F7")

# The single permitted accent pair, used only for pass/fail.
PASS_COLOUR = colors.HexColor("#1A7F45")
FAIL_COLOUR = colors.HexColor("#B4232A")

BODY_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"

# Pass/fail is carried by a glyph as well as by colour, so the document still
# reads correctly when printed in greyscale.
GLYPH_PASS = "\u2713"
GLYPH_FAIL = "\u2715"

PAGE_MARGIN = 20 * mm
CONTENT_WIDTH = A4[0] - 2 * PAGE_MARGIN


def _style(
    name: str,
    size: float,
    leading: Optional[float] = None,
    font: str = BODY_FONT,
    colour: colors.Color = INK,
    space_after: float = 0.0,
    space_before: float = 0.0,
    tracking: float = 0.0,
) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        fontName=font,
        fontSize=size,
        leading=leading if leading is not None else size * 1.45,
        textColor=colour,
        spaceAfter=space_after,
        spaceBefore=space_before,
        alignment=TA_LEFT,
        charSpace=tracking,
    )


STYLES = {
    "wordmark": _style("wordmark", 15, font=BOLD_FONT, tracking=1.6),
    "doc_title": _style("doc_title", 22, leading=26, font=BOLD_FONT, space_before=6),
    "doc_subtitle": _style("doc_subtitle", 9.5, colour=INK_MUTED, space_after=2),
    "section": _style(
        "section", 8, font=BOLD_FONT, colour=INK_MUTED,
        tracking=1.3, space_before=11, space_after=4,
    ),
    "body": _style("body", 9.2, leading=13.5),
    "body_muted": _style("body_muted", 9.2, leading=13.5, colour=INK_MUTED),
    "small": _style("small", 7.6, leading=11, colour=INK_MUTED),
    "score": _style("score", 46, leading=48, font=BOLD_FONT),
    "score_unit": _style("score_unit", 11, colour=INK_FAINT),
    "verdict": _style("verdict", 13, font=BOLD_FONT),
    "footer": _style("footer", 7, colour=INK_FAINT),
}


class HairRule(Flowable):
    """A 0.5 pt horizontal rule; used instead of table borders throughout."""

    def __init__(self, width: float, colour: colors.Color = RULE,
                 thickness: float = 0.5, space_before: float = 0.0):
        super().__init__()
        self.width = width
        self.colour = colour
        self.thickness = thickness
        self.space_before = space_before
        self.height = thickness + space_before

    def draw(self) -> None:
        self.canv.setStrokeColor(self.colour)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


# ---------------------------------------------------------------------------
# Spectrum chart
# ---------------------------------------------------------------------------
def render_spectrum_png(
    spectrum: Dict[str, Sequence[float]],
    bands: Sequence[Dict[str, Any]],
    *,
    width_in: float = 7.0,
    height_in: float = 2.45,
    dpi: int = 200,
) -> bytes:
    """Monochrome emission-vs-limit chart, matching the web app's chart styling."""
    frequency_mhz = [f / 1e6 for f in spectrum["frequency_hz"]]
    emission = list(spectrum["emission_dbuv"])
    limit = list(spectrum["limit_dbuv"])

    figure = Figure(figsize=(width_in, height_in), dpi=dpi, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)

    # Region above the limit line: shaded so an exceedance reads instantly even
    # in greyscale.
    axes.fill_between(
        frequency_mhz, limit, max(max(emission), max(limit)) + 12,
        facecolor="#0A0A0A", alpha=0.055, linewidth=0,
    )
    axes.plot(frequency_mhz, limit, color="#0A0A0A", linewidth=1.1,
              linestyle=(0, (5, 3)), label="Assumed limit (synthetic)")
    axes.plot(frequency_mhz, emission, color="#0A0A0A", linewidth=1.0,
              label="Simulated emission (max-hold)")

    # Band boundaries as hairlines with labels along the top.
    for band in bands[:-1]:
        axes.axvline(band["f_high_hz"] / 1e6, color="#D0D0D0",
                     linewidth=0.6, linestyle=(0, (1, 2)))

    axes.set_xscale("log")
    axes.set_xlim(min(frequency_mhz), max(frequency_mhz))
    axes.set_ylim(
        min(min(emission), min(limit)) - 6,
        max(max(emission), max(limit)) + 8,
    )
    axes.set_xlabel("Frequency (MHz)", fontsize=7.5, color="#4A4A4A")
    axes.set_ylabel("Level (dB\u00b5V)", fontsize=7.5, color="#4A4A4A")
    axes.tick_params(labelsize=7, colors="#4A4A4A", width=0.5, length=3)
    axes.grid(True, which="major", color="#EDEDED", linewidth=0.5)
    axes.grid(True, which="minor", color="#F6F6F6", linewidth=0.4)

    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color("#CFCFCF")
        axes.spines[spine].set_linewidth(0.6)

    legend = axes.legend(
        loc="lower left", fontsize=6.8, frameon=False, ncol=2,
        handlelength=2.6, borderaxespad=0.2,
    )
    for text in legend.get_texts():
        text.set_color("#4A4A4A")

    figure.tight_layout(pad=0.6)

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
_BASE_TABLE_STYLE: List[Tuple[Any, ...]] = [
    ("FONTNAME", (0, 0), (-1, 0), BOLD_FONT),
    ("FONTSIZE", (0, 0), (-1, -1), 8.4),
    ("TEXTCOLOR", (0, 0), (-1, 0), INK_MUTED),
    ("TEXTCOLOR", (0, 1), (-1, -1), INK),
    ("FONTNAME", (0, 1), (-1, -1), BODY_FONT),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE_STRONG),
    ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
]


def _configuration_table(parameter_display: Sequence[Dict[str, Any]]) -> Table:
    """Six design parameters as three rows of two label/value pairs.

    A single six-row column would not fit under the chart and the band table
    without spilling onto a second page; paired columns keep the whole assessment
    summary on page one, where it belongs.
    """
    pairs = list(parameter_display)
    half = (len(pairs) + 1) // 2
    left_column, right_column = pairs[:half], pairs[half:]

    rows: List[List[Any]] = []
    for index in range(half):
        left = left_column[index]
        right = right_column[index] if index < len(right_column) else None
        rows.append([
            left["label"], left["formatted"],
            right["label"] if right else "",
            right["formatted"] if right else "",
        ])

    column = CONTENT_WIDTH / 2.0
    table = Table(rows, colWidths=[
        column * 0.56, column * 0.44, column * 0.56, column * 0.44,
    ])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), BODY_FONT),
        ("FONTNAME", (1, 0), (1, -1), BOLD_FONT),
        ("FONTNAME", (3, 0), (3, -1), BOLD_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("TEXTCOLOR", (0, 0), (0, -1), INK_MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), INK_MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("TEXTCOLOR", (3, 0), (3, -1), INK),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("ALIGN", (3, 0), (3, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (1, 0), (1, -1), 20),
        ("RIGHTPADDING", (3, 0), (3, -1), 0),
        ("LEFTPADDING", (2, 0), (2, -1), 20),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]))
    return table


def _band_table(bands: Sequence[Dict[str, Any]]) -> Table:
    rows: List[List[Any]] = [[
        "Frequency band", "Peak", "Limit", "Margin", "Result",
    ]]
    style = list(_BASE_TABLE_STYLE)

    for index, band in enumerate(bands, start=1):
        passes = band["passes"]
        rows.append([
            band["label"],
            f"{band['simulated_peak_dbuv']:.1f} dB\u00b5V",
            f"{band['limit_at_peak_dbuv']:.1f} dB\u00b5V",
            f"{band['predicted_margin_db']:+.1f} dB",
            (GLYPH_PASS + "  PASS") if passes else (GLYPH_FAIL + "  FAIL"),
        ])
        style.append((
            "TEXTCOLOR", (4, index), (4, index),
            PASS_COLOUR if passes else FAIL_COLOUR,
        ))
        style.append(("FONTNAME", (4, index), (4, index), BOLD_FONT))

    table = Table(rows, colWidths=[
        CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.17, CONTENT_WIDTH * 0.17,
        CONTENT_WIDTH * 0.16, CONTENT_WIDTH * 0.20,
    ])
    table.setStyle(TableStyle(style))
    return table


def _score_block(result: Dict[str, Any]) -> Table:
    passes = result["verdict"] == "PASS"
    score = result["compliance_score"]
    verdict_colour = "#1A7F45" if passes else "#B4232A"
    verdict_text = (
        GLYPH_PASS + "  PREDICTED PASS" if passes
        else GLYPH_FAIL + "  PREDICTED FAIL"
    )

    left = [
        Paragraph(f"{score:.0f}<font size=16 color='#9A9A9A'> / 100</font>",
                  STYLES["score"]),
        Spacer(1, 3),
        Paragraph("Compliance score", STYLES["small"]),
    ]
    right = [
        Paragraph(
            f"<font color='{verdict_colour}'>{verdict_text}</font>",
            STYLES["verdict"],
        ),
        Spacer(1, 7),
        Paragraph(
            f"<b>{result['confidence_score']:.0f}/100</b> &nbsp;"
            f"{result['confidence_label']}",
            STYLES["body"],
        ),
        Spacer(1, 2),
        Paragraph("Verdict confidence", STYLES["small"]),
    ]

    table = Table(
        [[left, right]],
        colWidths=[CONTENT_WIDTH * 0.42, CONTENT_WIDTH * 0.58],
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LINEAFTER", (0, 0), (0, 0), 0.5, RULE),
        ("LEFTPADDING", (1, 0), (1, 0), 18),
    ]))
    return table


def _callout(title: str, body: str, *, wash: bool = True) -> Table:
    inner = [
        Paragraph(title, STYLES["section"]),
        Paragraph(body, STYLES["body"]),
    ]
    table = Table([[inner]], colWidths=[CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WASH if wash else PAPER),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LINEBEFORE", (0, 0), (0, -1), 1.6, INK),
    ]))
    return table


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------
def assessment_reference(result: Dict[str, Any]) -> str:
    """Short, stable reference derived from the configuration being assessed."""
    payload = "|".join(f"{k}={v}" for k, v in sorted(result["parameters"].items()))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8].upper()
    return f"CA-{digest}"


def _draw_page_furniture(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, _ = A4

    # Footer rule and text.
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(PAGE_MARGIN, PAGE_MARGIN - 6, width - PAGE_MARGIN, PAGE_MARGIN - 6)

    canvas.setFont(BODY_FONT, 7)
    canvas.setFillColor(INK_FAINT)
    canvas.drawString(
        PAGE_MARGIN, PAGE_MARGIN - 14,
        "CertifAI \u00b7 Virtual Pre-Compliance Assessment \u00b7 "
        "Not an accredited certification",
    )
    canvas.drawRightString(
        width - PAGE_MARGIN, PAGE_MARGIN - 14, f"Page {doc.page}"
    )
    canvas.restoreState()


def build_certificate(result: Dict[str, Any]) -> bytes:
    """Render one assessment result to a PDF and return the bytes."""
    reference = assessment_reference(result)
    generated = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    buffer = io.BytesIO()
    document = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
        title=f"Virtual Pre-Compliance Assessment {reference}",
        author="CertifAI",
        subject="Simulation-based EMC pre-compliance assessment",
    )
    frame = Frame(
        PAGE_MARGIN, PAGE_MARGIN, CONTENT_WIDTH,
        A4[1] - 2 * PAGE_MARGIN, id="body",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    document.addPageTemplates([
        PageTemplate(id="main", frames=[frame], onPage=_draw_page_furniture)
    ])

    story: List[Flowable] = []

    # --- Masthead ---------------------------------------------------------
    story.append(Paragraph("CERTIFAI", STYLES["wordmark"]))
    story.append(HairRule(CONTENT_WIDTH, RULE_STRONG, 0.8, space_before=6))
    story.append(Paragraph("Virtual Pre-Compliance Assessment", STYLES["doc_title"]))
    story.append(Paragraph(
        "Simulated conducted-emission screening for elevator drive and controller "
        "electronics, against a synthetic EN-12016-style limit line.",
        STYLES["doc_subtitle"],
    ))
    story.append(Spacer(1, 12))

    # --- Metadata ---------------------------------------------------------
    meta_rows = [
        ["Assessed configuration", result["device_name"]],
        ["Assessment reference", reference],
        ["Generated", generated],
        ["Model artifact", f"v{result['model_info']['artifact_version']}"],
    ]
    meta_table = Table(meta_rows, colWidths=[CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.70])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), BODY_FONT),
        ("FONTNAME", (1, 0), (1, -1), BOLD_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("TEXTCOLOR", (0, 0), (0, -1), INK_MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)

    # --- Headline result --------------------------------------------------
    story.append(HairRule(CONTENT_WIDTH, RULE, space_before=12))
    story.append(_score_block(result))
    story.append(HairRule(CONTENT_WIDTH, RULE))

    # --- Spectrum ---------------------------------------------------------
    # The chart comes directly after the verdict: the reader sees the conclusion,
    # then the evidence for it, before any of the tabulated detail.
    story.append(Paragraph("EMISSION SPECTRUM", STYLES["section"]))
    chart_aspect = 2.45 / 7.0
    chart_png = render_spectrum_png(result["spectrum"], result["bands"])
    story.append(Image(
        io.BytesIO(chart_png),
        width=CONTENT_WIDTH,
        height=CONTENT_WIDTH * chart_aspect,
    ))

    # --- Band results -----------------------------------------------------
    story.append(Paragraph("PER-BAND RESULT", STYLES["section"]))
    story.append(_band_table(result["bands"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Margin is the predicted headroom to the assumed limit line at the worst "
        "frequency in each band. Positive values indicate headroom. Peak levels are "
        "taken from the simulated max-hold trace with a 9 kHz resolution bandwidth.",
        STYLES["small"],
    ))

    # --- Configuration ----------------------------------------------------
    story.append(Paragraph("ASSESSED CONFIGURATION", STYLES["section"]))
    story.append(_configuration_table(result["parameter_display"]))

    # --- Risk factor ------------------------------------------------------
    # Page two carries the interpretation and the caveats. Breaking here is
    # deliberate rather than incidental, so neither page ends with a ragged gap.
    story.append(PageBreak())
    risk = result["top_risk_factor"]
    story.append(KeepTogether(_callout(
        "PRIMARY RISK FACTOR",
        f"<b>{risk['statement']}</b><br/><br/>{risk['suggestion']}",
    )))

    # --- Methodology ------------------------------------------------------
    consistency = result["model_info"]["simulation_consistency"]
    ablation = consistency.get("design_only_ablation") or {}
    story.append(Paragraph("METHOD AND VALIDATION", STYLES["section"]))

    method_body = (
        "The configuration above is simulated as a three-phase trapezoidal PWM "
        "switching waveform, from which the common-mode disturbance voltage at the "
        "power port is derived. That waveform is transformed by FFT, processed "
        "through a 9 kHz resolution-bandwidth receiver model, and compared against "
        "the assumed limit line. Gradient-boosted models then predict per-band "
        "pass/fail and margin from the design parameters together with the "
        "extracted spectral features. Those models carry monotonic constraints on "
        "every input, so they cannot learn a physically impossible relationship "
        "such as improved shielding increasing emissions."
    )
    story.append(Paragraph(method_body, STYLES["body"]))
    story.append(Spacer(1, 8))

    consistency_bits: List[str] = []
    if consistency.get("mean_balanced_accuracy") is not None:
        consistency_bits.append(
            f"mean balanced accuracy "
            f"{consistency['mean_balanced_accuracy'] * 100:.1f}%"
        )
    if consistency.get("mean_margin_mae_db") is not None:
        consistency_bits.append(
            f"mean margin error {consistency['mean_margin_mae_db']:.2f} dB"
        )
    consistency_text = (
        f"Simulation-consistency on {consistency.get('n_test_designs', 'held-out')} "
        f"held-out synthetic designs: {', '.join(consistency_bits)}."
        if consistency_bits else
        "Simulation-consistency metrics are recorded in the model's validation report."
    )
    if ablation.get("mean_balanced_accuracy") is not None:
        consistency_text += (
            f" Restricted to the six design parameters alone, with all spectral "
            f"features removed, the same models reach "
            f"{ablation['mean_balanced_accuracy'] * 100:.1f}% balanced accuracy and "
            f"{ablation['mean_margin_mae_db']:.2f} dB mean margin error."
        )
    consistency_text += (
        " These figures describe agreement with this tool's own physics simulation, "
        "not agreement with measured hardware."
    )
    story.append(Paragraph(consistency_text, STYLES["body_muted"]))

    # --- Assumptions and disclaimer --------------------------------------
    story.append(Paragraph("ASSUMPTIONS AND LIMITATIONS", STYLES["section"]))
    story.append(Paragraph(
        f"<b>Limit line is synthetic.</b> {result['limit_curve']['description']} "
        "The normative EN 12016 tables are copyrighted and not publicly "
        "redistributable, so a pass against this curve is not a statement about "
        "EN 12016 conformity.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>No measured data.</b> The simulator is calibrated to produce levels of "
        "the right order and the correct qualitative trends, but it is not fitted "
        "to hardware measurements. Absolute levels should be read as relative "
        "indicators for comparing design options, not as predicted lab readings.",
        STYLES["body"],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Conducted emissions only.</b> Radiated emissions, immunity, harmonics "
        "and functional-safety requirements are out of scope.",
        STYLES["body"],
    ))

    story.append(Spacer(1, 12))
    story.append(HairRule(CONTENT_WIDTH, RULE_STRONG, 0.8))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<b>Disclaimer.</b> {result['disclaimer_long']}",
        STYLES["body"],
    ))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def certificate_filename(result: Dict[str, Any]) -> str:
    reference = assessment_reference(result)
    slug = (
        "".join(
            character.lower() if character.isalnum() else "-"
            for character in result["device_name"]
        ).strip("-")
    )
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"certifai-pre-compliance-{slug}-{reference}.pdf"


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from pathlib import Path as _Path

    from devices import DEVICE_PROFILES
    from predictor import predict

    for profile in DEVICE_PROFILES[:2]:
        outcome = predict(
            profile.parameters, device_id=profile.id, device_name=profile.name
        )
        pdf = build_certificate(outcome)
        destination = _Path(__file__).parent / "models" / certificate_filename(outcome)
        destination.write_bytes(pdf)
        print(f"{destination.name}  ({len(pdf) / 1024:.0f} kB)")
