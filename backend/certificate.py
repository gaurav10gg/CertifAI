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

# The single permitted accent pair, used only for risk-tier extremes.
PASS_COLOUR = colors.HexColor("#1A7F45")
FAIL_COLOUR = colors.HexColor("#B4232A")
MODERATE_COLOUR = colors.HexColor("#0A0A0A")

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


def render_signals_png(
    signals: Sequence[Dict[str, Any]],
    *,
    width_in: float = 7.0,
    height_in: float = 3.6,
    dpi: int = 160,
) -> bytes:
    """Small-multiples of the five power-stage waveforms."""
    n = max(len(signals), 1)
    figure = Figure(figsize=(width_in, height_in), dpi=dpi, facecolor="white")
    FigureCanvasAgg(figure)
    axes_list = figure.subplots(n, 1, sharex=False)
    if n == 1:
        axes_list = [axes_list]
    for axes, trace in zip(axes_list, signals):
        axes.plot(trace["time_ms"], trace["values"], color="#0A0A0A", linewidth=0.8)
        axes.set_ylabel(trace.get("unit", ""), fontsize=6.5, color="#4A4A4A")
        axes.set_title(trace.get("label", ""), fontsize=7, loc="left", color="#0A0A0A", pad=2)
        axes.tick_params(labelsize=6, colors="#4A4A4A", width=0.4, length=2)
        axes.grid(True, color="#EDEDED", linewidth=0.4)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        axes.spines["left"].set_color("#CFCFCF")
        axes.spines["bottom"].set_color("#CFCFCF")
    if signals:
        axes_list[-1].set_xlabel("Time (ms)", fontsize=7, color="#4A4A4A")
    figure.tight_layout(pad=0.35, h_pad=0.4)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)
    return buffer.getvalue()


def render_shap_png(
    shap: Dict[str, Any],
    *,
    width_in: float = 7.0,
    height_in: float = 2.1,
    dpi: int = 160,
) -> bytes:
    rows = list(shap.get("contributions") or [])
    rows.sort(key=lambda row: abs(float(row["shap"])), reverse=True)
    # barh draws index 0 at the bottom; reverse so the largest |SHAP| is on top.
    labels = [row["label"] for row in reversed(rows)]
    values = [float(row["shap"]) for row in reversed(rows)]
    figure = Figure(figsize=(width_in, height_in), dpi=dpi, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    colours = ["#B4232A" if v > 0 else "#1A7F45" for v in values]
    axes.barh(labels, values, color=colours, height=0.55)
    axes.axvline(0, color="#0A0A0A", linewidth=0.6)
    axes.set_xlabel("Contribution to predicted margin (dB)", fontsize=7, color="#4A4A4A")
    axes.tick_params(labelsize=7, colors="#4A4A4A")
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    figure.tight_layout(pad=0.45)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)
    return buffer.getvalue()


def render_tradeoff_png(
    points: Sequence[Dict[str, Any]],
    current_khz: float,
    *,
    width_in: float = 7.0,
    height_in: float = 2.15,
    dpi: int = 160,
) -> bytes:
    """Compact EMC-score vs acoustic-risk frontier for the PDF report."""
    figure = Figure(figsize=(width_in, height_in), dpi=dpi, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)

    dominated = [p for p in points if not p.get("pareto_acoustic")]
    pareto = sorted(
        [p for p in points if p.get("pareto_acoustic")],
        key=lambda p: (float(p["emc_risk_score"]), float(p["acoustic_risk"])),
    )

    def _sizes(rows: Sequence[Dict[str, Any]], scale: float) -> List[float]:
        out: List[float] = []
        for row in rows:
            t = (float(row["switching_frequency_khz"]) - 3.0) / 13.0
            t = min(1.0, max(0.0, t))
            out.append((18.0 + 22.0 * t) * scale)
        return out

    if dominated:
        axes.scatter(
            [p["emc_risk_score"] for p in dominated],
            [p["acoustic_risk"] for p in dominated],
            s=_sizes(dominated, 0.55),
            facecolors="none",
            edgecolors="#9A9A9A",
            linewidths=0.7,
            zorder=2,
        )
    if pareto:
        axes.plot(
            [p["emc_risk_score"] for p in pareto],
            [p["acoustic_risk"] for p in pareto],
            color="#0A0A0A",
            linewidth=1.1,
            zorder=3,
        )
        axes.scatter(
            [p["emc_risk_score"] for p in pareto],
            [p["acoustic_risk"] for p in pareto],
            s=_sizes(pareto, 1.0),
            c="#0A0A0A",
            zorder=4,
        )

    current = min(
        points,
        key=lambda p: abs(float(p["switching_frequency_khz"]) - current_khz),
        default=None,
    )
    if current is not None:
        axes.scatter(
            [current["emc_risk_score"]],
            [current["acoustic_risk"]],
            s=90,
            facecolors="none",
            edgecolors="#0A0A0A",
            linewidths=1.3,
            zorder=5,
        )

    axes.set_xlabel("EMC risk score (higher = more headroom)", fontsize=7, color="#4A4A4A")
    axes.set_ylabel("Acoustic risk", fontsize=7, color="#4A4A4A")
    axes.tick_params(labelsize=6.5, colors="#4A4A4A", width=0.4, length=2)
    axes.grid(True, color="#EDEDED", linewidth=0.4, axis="y")
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    axes.spines["left"].set_color("#CFCFCF")
    axes.spines["bottom"].set_color("#CFCFCF")
    figure.tight_layout(pad=0.4)
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
        "Frequency band", "Peak", "Limit", "Margin", "Vs assumed limit",
    ]]
    style = list(_BASE_TABLE_STYLE)

    for index, band in enumerate(bands, start=1):
        clear = band["predicted_margin_db"] > 0
        rows.append([
            band["label"],
            f"{band['simulated_peak_dbuv']:.1f} dB\u00b5V",
            f"{band['limit_at_peak_dbuv']:.1f} dB\u00b5V",
            f"{band['predicted_margin_db']:+.1f} dB",
            (GLYPH_PASS + "  Clear") if clear else (GLYPH_FAIL + "  Exceeds"),
        ])
        style.append((
            "TEXTCOLOR", (4, index), (4, index),
            PASS_COLOUR if clear else FAIL_COLOUR,
        ))
        style.append(("FONTNAME", (4, index), (4, index), BOLD_FONT))

    table = Table(rows, colWidths=[
        CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.17, CONTENT_WIDTH * 0.17,
        CONTENT_WIDTH * 0.16, CONTENT_WIDTH * 0.20,
    ])
    table.setStyle(TableStyle(style))
    return table


def _score_block(result: Dict[str, Any]) -> Table:
    level = result.get("risk_level") or result.get("verdict") or "MODERATE"
    score = result.get("risk_score", result.get("compliance_score", 0))
    plus = result.get("risk_score_plus_minus")
    colour = {
        "LOW": "#1A7F45",
        "MODERATE": "#0A0A0A",
        "HIGH": "#B4232A",
    }.get(level, "#0A0A0A")
    label = result.get("risk_label", level.replace("_", " ").title())
    interval = f" \u00b1{plus:.0f}" if plus is not None else ""

    left = [
        Paragraph(
            f"{score:.0f}<font size=16 color='#9A9A9A'>{interval} / 100</font>",
            STYLES["score"],
        ),
        Spacer(1, 3),
        Paragraph("Risk score (higher = more headroom)", STYLES["small"]),
    ]
    right = [
        Paragraph(
            f"<font color='{colour}'>{label.upper()}</font>",
            STYLES["verdict"],
        ),
        Spacer(1, 7),
        Paragraph(
            result.get("risk_copy")
            or "Simulation-based EMC risk indicator, not a certification result.",
            STYLES["body"],
        ),
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
        "EMC Advisor \u00b7 Virtual EMC Pre-Compliance Report \u00b7 "
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
        title=f"Virtual EMC Pre-Compliance Report {reference}",
        author="EMC Advisor",
        subject="Simulation-based EMC pre-compliance risk assessment",
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
    story.append(Paragraph("EMC ADVISOR", STYLES["wordmark"]))
    story.append(HairRule(CONTENT_WIDTH, RULE_STRONG, 0.8, space_before=6))
    story.append(Paragraph("Virtual EMC Pre-Compliance Report", STYLES["doc_title"]))
    story.append(Paragraph(
        "Simulation-based EMC risk indicator for elevator drive electronics. "
        "Not a prediction of EN 12016 certification outcomes.",
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
    story.append(Paragraph(
        result.get("framing")
        or "This tool estimates EMC risk from simulated physics. It does not "
        "predict EN 12016 certification outcomes.",
        STYLES["small"],
    ))
    story.append(Spacer(1, 8))

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
        "PRIMARY RISK DRIVER",
        f"<b>{risk['statement']}</b><br/><br/>{risk['suggestion']}",
    )))

    shap = result.get("shap")
    if shap and shap.get("contributions"):
        story.append(Paragraph("WHY THIS MARGIN", STYLES["section"]))
        story.append(Paragraph(
            "Exact attribution for this specific configuration.",
            STYLES["small"],
        ))
        shap_png = render_shap_png(shap)
        story.append(Image(io.BytesIO(shap_png), width=CONTENT_WIDTH, height=CONTENT_WIDTH * 2.1 / 7.0))
        story.append(Paragraph(shap.get("note", ""), STYLES["small"]))

    measures = result.get("countermeasures") or []
    if shap and shap.get("contributions") and measures:
        story.append(Paragraph(
            "These lists can diverge: the chart explains why the current design "
            "sits where it does, including factors already optimized (like "
            "shielding). The recommendations below only suggest parameters with "
            "room left to improve.",
            STYLES["small"],
        ))
        story.append(Spacer(1, 6))

    if measures:
        story.append(Paragraph("WHAT TO CHANGE NEXT", STYLES["section"]))
        story.append(Paragraph(
            "Ranked by remaining headroom to improve, excluding parameters "
            "already near their safe limit.",
            STYLES["small"],
        ))
        for index, item in enumerate(measures[:3], start=1):
            story.append(Paragraph(
                f"<b>{index}. {item['label']}</b> \u2014 {item['suggestion']}",
                STYLES["body"],
            ))
            story.append(Spacer(1, 4))

    try:
        from simulate import DeviceParameters as _TradeoffParams
        from tradeoff import run_tradeoff as _run_tradeoff

        sweep = _run_tradeoff(
            _TradeoffParams(**result["parameters"])
        )
        story.append(Paragraph("DESIGN TRADE-OFF CONTEXT", STYLES["section"]))
        story.append(Paragraph(
            "The switching-frequency recommendation above is not free. "
            "Audible noise and motor current ripple both rise as the carrier "
            "is lowered to improve EMC margin. The chart is EMC headroom "
            "against acoustic risk; ripple traces the same Pareto set.",
            STYLES["small"],
        ))
        tradeoff_png = render_tradeoff_png(
            sweep["points"],
            float(result["parameters"]["switching_frequency_khz"]),
        )
        story.append(Image(
            io.BytesIO(tradeoff_png),
            width=CONTENT_WIDTH,
            height=CONTENT_WIDTH * 2.15 / 7.0,
        ))
        story.append(Paragraph(sweep["caption"], STYLES["small"]))
        story.append(Spacer(1, 6))
    except Exception:
        pass

    pq = result.get("power_quality") or []
    if pq:
        story.append(Paragraph("POWER QUALITY (THD)", STYLES["section"]))
        pq_rows: List[List[Any]] = [["Signal", "THD", "Dominant harmonics"]]
        for report in pq:
            pq_rows.append([
                report["label"],
                f"{report['thd_percent']:.1f} %",
                report["dominant_statement"],
            ])
        pq_table = Table(pq_rows, colWidths=[
            CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.14, CONTENT_WIDTH * 0.64,
        ])
        pq_table.setStyle(TableStyle(_BASE_TABLE_STYLE + [
            ("ALIGN", (2, 0), (2, -1), "LEFT"),
        ]))
        story.append(pq_table)
        story.append(Paragraph(
            "THD is a power-quality metric on motor and input current. It is not "
            "an EMC conducted-emission result and is not scored against EN 12016.",
            STYLES["small"],
        ))

    signals = result.get("signals") or []
    if signals:
        story.append(Paragraph("POWER-STAGE WAVEFORMS", STYLES["section"]))
        sig_png = render_signals_png(signals)
        story.append(Image(
            io.BytesIO(sig_png),
            width=CONTENT_WIDTH,
            height=CONTENT_WIDTH * 3.6 / 7.0,
        ))

    # --- Methodology ------------------------------------------------------
    consistency = result["model_info"]["simulation_consistency"]
    ablation = consistency.get("design_only_ablation") or {}
    story.append(Paragraph("METHOD AND VALIDATION", STYLES["section"]))

    method_body = (
        "The configuration is simulated as a three-phase trapezoidal PWM switching "
        "waveform. From the three pole voltages the model forms the common-mode "
        "voltage that drives conducted emissions, together with DC-link, motor "
        "voltage, motor current and rectifier input current for power-quality "
        "inspection. The common-mode record is transformed by FFT, processed "
        "through a 9 kHz resolution-bandwidth receiver model, and compared against "
        "the assumed limit line. Gradient-boosted models then predict per-band "
        "margin from the design parameters together with the extracted spectral "
        "features. Those models carry monotonic constraints on every input, so "
        "they cannot learn a physically impossible relationship such as improved "
        "shielding increasing emissions. The headline figure is a risk score "
        "(higher = more simulated headroom), summarised as low / moderate / high "
        "risk \u2014 not a pass/fail certification prediction."
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
        "redistributable, so sitting below this curve is not a statement about "
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
        "<b>Conducted emissions and power quality are separate.</b> The risk score "
        "is driven by common-mode conducted emissions. THD figures describe motor "
        "and input current distortion and are not EMC limit comparisons. Radiated "
        "emissions, immunity and functional-safety requirements are out of scope.",
        STYLES["body"],
    ))

    story.append(Spacer(1, 12))
    story.append(HairRule(CONTENT_WIDTH, RULE_STRONG, 0.8))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<b>Disclaimer.</b> {result['disclaimer_long']}",
        STYLES["body"],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Guidance reviewed against publicly discussed elevator-drive EMC practice "
        "(KONE mentor feedback, September 2026). That review informed the risk "
        "framing and the multi-signal scope; it is not a product endorsement and "
        "does not constitute validation against KONE hardware.",
        STYLES["small"],
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
    return f"emc-advisor-pre-compliance-{slug}-{reference}.pdf"


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
