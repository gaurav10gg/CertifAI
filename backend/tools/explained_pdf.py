"""
A plain-language PDF of the whole EMC Advisor: every physics term defined
with an analogy, the full pipeline, and an honest list of what the tool is not.

Run from the backend folder:

    .venv\\Scripts\\python.exe tools\\explained_pdf.py

Writes ``docs/EMC_Advisor_Explained.pdf``.
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "docs" / "EMC_Advisor_Explained.pdf"

INK = colors.HexColor("#0A0A0A")
INK_MUTED = colors.HexColor("#6B6B6B")
INK_FAINT = colors.HexColor("#9A9A9A")
RULE = colors.HexColor("#E5E5E5")
RULE_STRONG = colors.HexColor("#C9C9C9")
WASH = colors.HexColor("#F7F7F7")
PASS_COLOUR = colors.HexColor("#1A7F45")
FAIL_COLOUR = colors.HexColor("#B4232A")

BODY_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"
PAGE_MARGIN = 18 * mm
PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 2 * PAGE_MARGIN


def _style(
    name: str,
    size: float,
    leading: float | None = None,
    font: str = BODY_FONT,
    colour: colors.Color = INK,
    space_after: float = 0,
    space_before: float = 0,
    tracking: float = 0,
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
    "kicker": _style("kicker", 8, font=BOLD_FONT, colour=INK_MUTED, tracking=1.4),
    "cover_title": _style("cover_title", 26, leading=31, font=BOLD_FONT, space_before=4),
    "cover_sub": _style("cover_sub", 11.2, leading=16, colour=INK_MUTED, space_before=8),
    "part": _style(
        "part", 8, font=BOLD_FONT, colour=INK_MUTED, tracking=1.4,
        space_before=14, space_after=3,
    ),
    "h1": _style("h1", 14, leading=18, font=BOLD_FONT, space_before=2, space_after=6),
    "h2": _style("h2", 11, leading=14, font=BOLD_FONT, space_before=10, space_after=4),
    "body": _style("body", 9.4, leading=13.6, space_after=7),
    "muted": _style("muted", 9.4, leading=13.6, colour=INK_MUTED, space_after=7),
    "small": _style("small", 7.7, leading=11, colour=INK_MUTED, space_after=4),
    "caption": _style(
        "caption", 7.4, leading=10.4, colour=INK_FAINT, space_after=10, space_before=2,
    ),
    "th": _style("th", 7.4, leading=10, font=BOLD_FONT, colour=INK_FAINT),
    "td": _style("td", 8.1, leading=11.2),
    "box_title": _style("box_title", 8, font=BOLD_FONT, tracking=0.5, space_after=3),
    "box_body": _style("box_body", 8.6, leading=12.3),
    "term": _style("term", 9.4, leading=12.4, font=BOLD_FONT, space_after=1, space_before=6),
    "def": _style("def", 9.2, leading=13.2, space_after=4),
}


class HairRule(Flowable):
    def __init__(self, width: float, colour: colors.Color = RULE, thickness: float = 0.5):
        super().__init__()
        self.width = width
        self.colour = colour
        self.thickness = thickness
        self.height = thickness

    def draw(self) -> None:
        self.canv.setStrokeColor(self.colour)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def bullets(items: Sequence[str], font_size: float = 9.2) -> ListFlowable:
    style = ParagraphStyle(
        "bullet",
        fontName=BODY_FONT,
        fontSize=font_size,
        leading=font_size * 1.42,
        textColor=INK,
    )
    return ListFlowable(
        [ListItem(Paragraph(item, style), leftIndent=12, bulletColor=INK) for item in items],
        bulletType="bullet",
        start="disc",
        leftIndent=14,
        bulletFontName=BODY_FONT,
        bulletFontSize=font_size,
        spaceAfter=8,
    )


def callout(title: str, body: str, kind: str = "note") -> KeepTogether:
    bar = {"honesty": FAIL_COLOUR, "ok": PASS_COLOUR, "note": INK}[kind]
    inner = Table(
        [[P(title, "box_title")], [P(body, "box_body")]],
        colWidths=[CONTENT_W - 8 * mm],
    )
    inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (0, 0), 7),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), WASH),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    wrapper = Table([[inner]], colWidths=[CONTENT_W])
    wrapper.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, 0), 2.2, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return KeepTogether([wrapper, Spacer(1, 9)])


def simple_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    widths: Sequence[float] | None = None,
) -> Table:
    col_w = list(widths) if widths else [CONTENT_W / len(headers)] * len(headers)
    data = [[P(h, "th") for h in headers]]
    for row in rows:
        data.append([P(cell, "td") for cell in row])
    table = Table(data, colWidths=col_w, repeatRows=1)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE_STRONG),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
    ]))
    return table


def png_image(png: bytes, width: float) -> Image:
    from reportlab.lib.utils import ImageReader

    reader = ImageReader(io.BytesIO(png))
    _iw, ih = reader.getSize()
    iw = float(_iw)
    return Image(io.BytesIO(png), width=width, height=width * (ih / iw))


def _finish_fig(fig: Figure) -> bytes:
    fig.tight_layout(pad=0.35)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor="white")
    return buf.getvalue()


def figure_power_path() -> bytes:
    fig = Figure(figsize=(7.2, 1.85), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    boxes = [
        (0.15, "Mains\n50 Hz"),
        (1.85, "Rectifier\nturns AC to DC"),
        (3.55, "DC link\n565 V tank"),
        (5.25, "Inverter\nswitches fast"),
        (6.95, "Cable\nto the motor"),
        (8.55, "Motor\nmoves the car"),
    ]
    for x, label in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, 0.55), 1.45, 1.15,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#F7F7F7", edgecolor="#0A0A0A", linewidth=1.0,
        ))
        ax.text(x + 0.725, 1.12, label, ha="center", va="center", fontsize=7.2, color="#0A0A0A")
        if x < 8:
            ax.annotate(
                "",
                xy=(x + 1.55, 1.12),
                xytext=(x + 1.45, 1.12),
                arrowprops={"arrowstyle": "->", "color": "#0A0A0A", "lw": 1.0},
            )
    ax.text(
        5.0, 0.18,
        "Electricity walks through the building like this. EMC Advisor watches the inverter and the cable.",
        ha="center", fontsize=7, color="#6B6B6B",
    )
    return _finish_fig(fig)


def figure_pwm() -> bytes:
    fig = Figure(figsize=(7.2, 2.35), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    t = [i / 400.0 for i in range(400)]
    sine = [0.55 + 0.42 * math.sin(2 * math.pi * 2.2 * x) for x in t]
    pwm = []
    for x in t:
        cycle = (x * 18.0) % 1.0
        pwm.append(1.05 if cycle < 0.55 else 0.08)
    ax.plot(t, [s + 1.15 for s in sine], color="#9A9A9A", linewidth=1.3, label="Smooth wall wave")
    ax.plot(t, pwm, color="#0A0A0A", linewidth=1.2, label="PWM from the inverter")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.42, 2.35)
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=7.5, color="#4A4A4A")
    ax.tick_params(labelsize=7, colors="#4A4A4A")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CFCFCF")
    return _finish_fig(fig)


def figure_bands() -> bytes:
    freqs = [0.15, 0.5, 5.0, 30.0]
    limits = [79.0, 73.0, 73.0, 70.0]
    fig = Figure(figsize=(7.2, 2.4), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(freqs, limits, color="#0A0A0A", linewidth=1.6)
    ax.scatter(freqs, limits, color="#0A0A0A", s=18, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.12, 35)
    ax.set_ylim(66, 84)
    ax.set_xticks([0.15, 0.5, 5, 30])
    ax.set_xticklabels(["0.15", "0.5", "5", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Frequency (MHz)  —  how fast the wiggle is", fontsize=7.5, color="#4A4A4A")
    ax.set_ylabel("Assumed limit (dBuV)", fontsize=7.5, color="#4A4A4A")
    ax.tick_params(labelsize=7, colors="#4A4A4A")
    ax.axvspan(0.15, 0.5, color="#0A0A0A", alpha=0.05)
    ax.axvspan(0.5, 5, color="#0A0A0A", alpha=0.02)
    ax.axvspan(5, 30, color="#0A0A0A", alpha=0.06)
    ax.text(0.27, 67.2, "Band A\n150–500 kHz", fontsize=6.8, color="#6B6B6B", ha="center")
    ax.text(1.6, 67.2, "Band B\n0.5–5 MHz", fontsize=6.8, color="#6B6B6B", ha="center")
    ax.text(12, 67.2, "Band C\n5–30 MHz", fontsize=6.8, color="#6B6B6B", ha="center")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _finish_fig(fig)


def figure_score() -> bytes:
    margins = [i / 10.0 for i in range(-160, 161)]
    scores = [max(0.0, min(100.0, 50.0 + 50.0 * math.tanh(m / 8.0))) for m in margins]
    fig = Figure(figsize=(7.2, 2.3), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(margins, scores, color="#0A0A0A", linewidth=1.6)
    ax.axhline(70, color="#1A7F45", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.axhline(40, color="#B4232A", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.axvline(0, color="#C9C9C9", linewidth=0.7)
    ax.text(4.2, 88, "LOW  (score 70-100)", fontsize=7, color="#1A7F45")
    ax.text(-10.8, 52, "MODERATE  (40-69)", fontsize=7, color="#0A0A0A")
    ax.text(-15.2, 10, "HIGH  (below 40)", fontsize=7, color="#B4232A")
    ax.set_xlim(-16, 16)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Margin: how far below (right) or above (left) the assumed limit, in dB", fontsize=7.5, color="#4A4A4A")
    ax.set_ylabel("Band score", fontsize=7.5, color="#4A4A4A")
    ax.tick_params(labelsize=7, colors="#4A4A4A")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _finish_fig(fig)


def figure_tradeoff() -> bytes:
    """Closed-form proxies vs frequency — the same shapes the app uses."""
    fs = [3.0 + i * 0.5 for i in range(27)]
    acoustic = [100.0 / (1.0 + math.exp((f - 8.0) / 2.5)) for f in fs]
    # Ripple cost: 100 * tanh(V/(4 L f) / 5), V=565, L=0.006
    ripple = []
    for f in fs:
        di = 565.0 / (4.0 * 0.006 * f * 1e3)
        ripple.append(100.0 * math.tanh(di / 5.0))
    fig = Figure(figsize=(7.2, 2.45), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(fs, acoustic, color="#0A0A0A", linewidth=1.6, label="Acoustic risk (whine)")
    ax.plot(fs, ripple, color="#0A0A0A", linewidth=1.4, linestyle=(0, (4, 2)), label="Ripple cost (jerky current)")
    ax.set_xlim(3, 16)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Switching frequency (kHz)", fontsize=7.5, color="#4A4A4A")
    ax.set_ylabel("0–100 cost", fontsize=7.5, color="#4A4A4A")
    ax.tick_params(labelsize=7, colors="#4A4A4A")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.annotate(
        "Lower carrier: quieter EMC, louder cabin, more ripple",
        xy=(4.0, 83),
        xytext=(6.2, 92),
        fontsize=6.8,
        color="#6B6B6B",
        arrowprops={"arrowstyle": "->", "color": "#9A9A9A", "lw": 0.7},
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _finish_fig(fig)


def figure_pipeline() -> bytes:
    fig = Figure(figsize=(7.2, 1.55), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    steps = [
        (0.2, "1. Knobs\n(your design)"),
        (2.15, "2. Physics\nsimulator"),
        (4.1, "3. Measure\nthe spectrum"),
        (6.05, "4. ML models\nscore the risk"),
        (8.0, "5. Report\nand trade-offs"),
    ]
    for x, label in steps:
        ax.add_patch(FancyBboxPatch(
            (x, 0.28), 1.7, 1.05,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#F7F7F7", edgecolor="#0A0A0A", linewidth=1.0,
        ))
        ax.text(x + 0.85, 0.80, label, ha="center", va="center", fontsize=7.2, color="#0A0A0A")
        if x < 7:
            ax.add_patch(FancyArrowPatch(
                (x + 1.72, 0.80), (x + 2.12, 0.80),
                arrowstyle="-|>", mutation_scale=8, color="#0A0A0A", lw=1.0,
            ))
    return _finish_fig(fig)


def on_page(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFillColor(INK_FAINT)
    canvas.setFont(BODY_FONT, 7)
    if doc.page > 1:
        canvas.drawString(PAGE_MARGIN, PAGE_H - 12 * mm, "EMC Advisor  ·  explained from zero")
        canvas.drawRightString(PAGE_W - PAGE_MARGIN, PAGE_H - 12 * mm, "Not a certification document")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(PAGE_MARGIN, PAGE_H - 13.2 * mm, PAGE_W - PAGE_MARGIN, PAGE_H - 13.2 * mm)
        canvas.line(PAGE_MARGIN, 12.5 * mm, PAGE_W - PAGE_MARGIN, 12.5 * mm)
        canvas.drawString(
            PAGE_MARGIN, 8.5 * mm,
            "A simulation-based risk indicator. It does not predict EN 12016 outcomes.",
        )
        canvas.drawRightString(PAGE_W - PAGE_MARGIN, 8.5 * mm, str(doc.page))
    canvas.restoreState()


def build() -> bytes:
    story: List[Flowable] = []

    # ------------------------------------------------------------------ cover
    story.append(Spacer(1, 10 * mm))
    story.append(P("EMC ADVISOR  ·  PLAIN-LANGUAGE GUIDE", "kicker"))
    story.append(HairRule(CONTENT_W, INK, 1.1))
    story.append(P(
        "What we built, and what<br/>every physics word means.",
        "cover_title",
    ))
    story.append(P(
        "A complete tour of the Virtual EMC Pre-Compliance Advisor. "
        "Every technical word is introduced with a picture or an everyday analogy "
        "before it is used again. You do not need to be an engineer to finish this "
        "document. You only need to be curious about why an elevator motor can "
        "make a radio angry, and how a computer can warn you before anyone books "
        "a test chamber.",
        "cover_sub",
    ))
    story.append(Spacer(1, 8))
    story.append(callout(
        "THE WHOLE PRODUCT, IN ONE STORY A CHILD CAN KEEP",
        "An elevator is a room that a motor lifts. The motor wants a smooth, "
        "changeable heartbeat of electricity. The wall socket only offers a "
        "fixed hum. A box called a <b>drive</b> rebuilds that hum by slamming "
        "tiny electronic taps on and off thousands of times a second. Sharp, "
        "fast slams leave leftover wiggles that travel along the motor cable "
        "and can bother radios and other machines in the building. Those leftover "
        "wiggles are <b>electromagnetic noise</b>. EMC Advisor is a practice test "
        "on a computer: it pretends to listen to that noise, scores how risky "
        "the design looks, explains which knobs made it better or worse, and "
        "shows that making the slams slower (quieter noise) also makes the cabin "
        "whine more and the motor current wobble more. It is a warning light, "
        "not a medal from a real test room.",
        "ok",
    ))
    story.append(simple_table(
        ["", ""],
        [
            ["What this is", "A computer tool that estimates electromagnetic noise risk for elevator drives, from a handful of design choices."],
            ["What this is not", "A laboratory certificate, a prediction of EN 12016, or a replacement for measured hardware."],
            ["Who it is for", "Anyone who needs to understand the product: engineers, managers, judges, first-time readers, and curious children."],
            ["How to read it", "Start at page 2. Skip to the glossary at the end whenever a word feels sticky."],
        ],
        [CONTENT_W * 0.22, CONTENT_W * 0.78],
    ))
    story.append(Spacer(1, 6))
    story.append(simple_table(
        ["Part", "Question it answers"],
        [
            ["0", "How do I read this, and what do the units mean?"],
            ["1", "What is an elevator drive, and why can it make radio noise?"],
            ["2", "Which knobs can a designer actually turn?"],
            ["3", "How does the computer pretend to be a laboratory?"],
            ["4", "How does a gap in decibels become a score, and why use machine learning?"],
            ["5", "If slower switching helps noise, what does it cost?"],
            ["6", "What does each screen in the app actually show?"],
            ["7", "What must we never claim?"],
            ["8", "Every specialised word, in one breath each."],
        ],
        [CONTENT_W * 0.12, CONTENT_W * 0.88],
    ))
    story.append(Spacer(1, 8))
    story.append(callout(
        "THE PROMISE OF THIS DOCUMENT",
        "If a word is important, it is defined the first time it appears, in ordinary "
        "language, with a picture when a picture helps. Later chapters reuse the "
        "word without repeating the whole lecture. Nothing is left as “you would "
        "know this if you were an EMC engineer.”",
        "ok",
    ))

    story.append(PageBreak())

    # ------------------------------------------------------------------ how to read
    story.append(P("PART 0", "part"))
    story.append(P("How to read this", "h1"))
    story.append(P(
        "Imagine you have never heard the letters E-M-C. That is the intended "
        "starting point. Each chapter answers one question. Analogies are tools, "
        "not jokes: when we say “a tap turning on and off,” we mean that is what "
        "the inverter is doing to voltage, thousands of times a second."
    ))
    story.append(P(
        "A few units appear often. You do not need to memorise them. Keep this "
        "tiny table nearby:"
    ))
    story.append(simple_table(
        ["Unit", "What it measures", "Everyday feel"],
        [
            ["Volt (V)", "Electrical “push”", "A wall socket in Europe is about 230 V. The drive’s internal tank is 565 V."],
            ["Ampere (A)", "How much charge is flowing", "A kettle is a few amps. A mid-size elevator motor might be 45 A."],
            ["Hertz (Hz)", "How many times something repeats in one second", "Mains electricity hums at 50 Hz. A piano’s middle A is 440 Hz."],
            ["Kilohertz (kHz)", "A thousand hertz", "The drive’s switching “drumbeat” is 3 to 16 kHz — right at the top of human hearing."],
            ["Megahertz (MHz)", "A million hertz", "FM radio lives around 100 MHz. Our EMC bands stop at 30 MHz."],
            ["Decibel (dB)", "A comparison of two sizes, on a compressed scale", "Plus 6 dB is roughly “twice as big.” Minus 6 dB is “half.”"],
        ],
        [CONTENT_W * 0.18, CONTENT_W * 0.32, CONTENT_W * 0.50],
    ))
    story.append(P(
        "When a number is a <b>simulation</b>, we will say so. A simulation is a "
        "story the computer tells using equations. It can be a very good story. "
        "It is still a story, not a photograph from a test lab.",
        "muted",
    ))

    # ------------------------------------------------------------------ part 1
    story.append(P("PART 1", "part"))
    story.append(P("The problem in the real world", "h1"))
    story.append(P("What is an elevator drive?", "h2"))
    story.append(P(
        "An elevator car is a heavy box. A motor must lift it smoothly, stop it "
        "exactly, and not bounce the passengers. The motor wants a special kind "
        "of electricity: a rotating push whose speed we can change. The wall "
        "socket does not supply that. It supplies a fixed 50 Hz wave."
    ))
    story.append(P(
        "The <b>drive</b> (also called a VFD, “variable-frequency drive”) sits "
        "between the wall and the motor. It is a box of power electronics. Its "
        "job is to take wall electricity and rebuild it into whatever speed and "
        "strength the motor needs right now."
    ))
    story.append(png_image(figure_power_path(), CONTENT_W))
    story.append(P(
        "Figure. Electricity’s walk from the wall to the motor. EMC Advisor "
        "cares most about the inverter and the long cable.",
        "caption",
    ))

    story.append(P("Switching: a tap that slams", "h2"))
    story.append(P(
        "The clever trick inside the inverter is called <b>switching</b>. Instead "
        "of trying to sculpt a perfect smooth wave with expensive analogue parts, "
        "the inverter has electronic taps (transistors) that slam fully on or "
        "fully off, thousands of times a second. The average of those slams, over "
        "a short time, looks like a smooth wave to the slow, heavy motor. The "
        "motor is happy. The radio in the next room may not be."
    ))
    story.append(P(
        "That on–off pattern is <b>PWM</b> — pulse-width modulation. “Pulse” means "
        "a burst of voltage. “Width” means how long the burst stays on. By "
        "changing the width, the inverter changes the average voltage the motor "
        "feels, without ever sitting at a half-on setting that would waste heat."
    ))
    story.append(png_image(figure_pwm(), CONTENT_W))
    story.append(P(
        "Figure. The grey curve is the kind of smooth wave the wall supplies. "
        "The black stairs are PWM: hard on, hard off. The motor averages the "
        "stairs. The radio hears the sharp corners.",
        "caption",
    ))

    story.append(P("Why sharp corners make radio noise", "h2"))
    story.append(P(
        "A smooth wave is one note. A square slam is a chord: the main note, "
        "plus a stack of higher notes called <b>harmonics</b>. Those extra notes "
        "are not music. They are leftover energy at frequencies where radios, "
        "sensors, and neighbouring electronics also live."
    ))
    story.append(P(
        "The faster the tap slams (a steep edge), the more high notes appear. "
        "The more often it slams (a higher switching frequency), the more of "
        "those notes fall into the band a test laboratory actually measures. "
        "That is the whole EMC problem in two sentences."
    ))

    story.append(P("EMC, in one picture in your head", "h2"))
    story.append(P(
        "<b>Electromagnetic compatibility (EMC)</b> is a politeness rule for "
        "machines. Every electrical thing is allowed to make a little invisible "
        "weather around itself. It is not allowed to make a storm that ruins "
        "its neighbours. “Compatible” means: your elevator still works when "
        "someone’s radio is on, and their radio still works when your elevator "
        "moves."
    ))
    story.append(P(
        "There are two directions. <b>Emissions</b> are the weather you send out. "
        "<b>Immunity</b> is how well you shrug off other people’s weather. This "
        "tool only estimates <b>conducted emissions</b>: noise that sneaks out "
        "along the wires, not the noise that flies through the air as a radio "
        "wave (that would be <b>radiated</b> emissions, which we do not model)."
    ))
    story.append(callout(
        "WHY ELEVATORS CARE",
        "An elevator sits in a building full of other electronics: fire alarms, "
        "door controllers, hearing loops, radios, medical equipment in a hospital "
        "tower. A noisy drive can make a neighbour misbehave. The usual way to "
        "check this is a booked EMC laboratory. Those bookings are slow and "
        "expensive. EMC Advisor exists so a designer can ask “is this cable and "
        "this carrier likely to be a problem?” before anyone ships a cabinet.",
        "note",
    ))

    # ------------------------------------------------------------------ part 2
    story.append(P("PART 2", "part"))
    story.append(P("The six knobs you can turn", "h1"))
    story.append(P(
        "The whole assessment starts from seven numbers a designer already knows "
        "(or can guess). Six of them feed the EMC models. The seventh only "
        "changes power quality. None of them is a photograph of a real drive. "
        "They are the levers the physics in the box responds to."
    ))
    story.append(simple_table(
        ["Knob", "Plain meaning", "If you turn it this way…"],
        [
            [
                "Switching frequency<br/>3–16 kHz",
                "How often the electronic taps slam, in thousands of times per second. Also called the <b>carrier</b>.",
                "Higher: smoother motor current, quieter cabin, <b>more</b> EMC noise in the test bands. Lower: the opposite.",
            ],
            [
                "dv/dt<br/>500–10 000 V per microsecond",
                "How steep each slam is. “dV/dt” is maths for “how many volts the edge climbs in a millionth of a second.”",
                "Steeper: less switching heat, <b>more</b> high-frequency hiss (especially Band C). Gentler: the reverse.",
            ],
            [
                "Motor cable length<br/>1–120 m",
                "How long the wire from the drive to the motor is. A high-rise shaft can need 85 m.",
                "Longer: more invisible capacitance to earth, more common-mode current, usually more emissions.",
            ],
            [
                "Shielding quality<br/>0–1",
                "How well the cable’s metal braid and the drive’s filters bottle the noise. 1 is a very good screen.",
                "Better shield: less noise escapes. This is often the biggest single lever.",
            ],
            [
                "Load current<br/>5–200 A",
                "How hard the motor is working. More current is a bigger machine, or a heavier car.",
                "More current: a little more noise in this model, not twice the current and twice the noise.",
            ],
            [
                "PWM type",
                "The pattern of the slams: ordinary sine PWM, space-vector, discontinuous, or randomised.",
                "Discontinuous PWM slams less often, so it usually emits a little less. Random PWM smears tones.",
            ],
            [
                "Input filter quality<br/>0–1",
                "A choke on the incoming mains. It smooths the current the building sees.",
                "This is a <b>power-quality</b> knob only. It is not allowed to pretend it fixes motor-cable EMC.",
            ],
        ],
        [CONTENT_W * 0.22, CONTENT_W * 0.38, CONTENT_W * 0.40],
    ))
    story.append(P(
        "Two extra ideas live inside those knobs and deserve their own sentences.",
        "muted",
    ))
    story.append(P("<b>Common-mode voltage.</b> The inverter has three output wires, one per motor phase. If you average those three voltages, you do not get zero. You get a staircase that jumps six times per switching cycle. That average is <b>common-mode voltage</b>. It is the invisible push that drives noise current into the earth through the cable’s capacitance. It is not the same as the useful voltage between two motor wires (that useful one is <b>differential</b>).", "def"))
    story.append(P("<b>Parasitic capacitance.</b> Two metal things near each other, with insulation in between, act a bit like a bucket that can be filled with charge. The motor cable and the building earth are such a pair. We do not install that capacitor on purpose, so it is called parasitic: an unwanted guest. Longer cable, bigger guest. When the common-mode voltage slams, current i = C x dv/dt rushes into that guest and out through the earth — that rush is a lot of the EMC problem.", "def"))

    # ------------------------------------------------------------------ part 3
    story.append(P("PART 3", "part"))
    story.append(P("How the computer pretends to be a laboratory", "h1"))
    story.append(png_image(figure_pipeline(), CONTENT_W))
    story.append(P(
        "Figure. The five-step pipeline. A click in the app always starts at "
        "step 1 and finishes at step 5. Nothing is a lookup table of old tests.",
        "caption",
    ))

    story.append(P("The physics simulator", "h2"))
    story.append(P(
        "For each set of knobs, the computer builds a fake but physically "
        "honest movie of the voltages and currents. It does not invent random "
        "wiggly lines. It follows a chain that power-electronics textbooks "
        "have used for decades:"
    ))
    story.append(bullets([
        "<b>Trapezoid edges.</b> Each slam is a ramp, not an infinitely sharp cliff. The ramp’s slope is your dv/dt. Sharp ramps put energy higher up in frequency.",
        "<b>Common-mode staircase.</b> Average the three legs. That staircase is what drives conducted noise.",
        "<b>Cable as a sponge and a guitar string.</b> The sponge is the capacitance to earth. The guitar string is resonance: a long cable rings at a pitch set by its length, and at odd multiples of that pitch (3x, 5x), like a pipe that sounds overtones.",
        "<b>Shield as a lid.</b> A good braid plus a common-mode choke lets less of the ringing out, especially at high frequency.",
        "<b>A little random floor.</b> Real instruments are never silent. A small mixed white-and-pink noise floor is added so the picture is not cartoon-clean. The random draw is locked to the design, so the same knobs always give the same report.",
    ]))
    story.append(P(
        "A real EMC receiver does not take one tiny snapshot. It <b>dwells</b> "
        "on each frequency and remembers the worst moment (max-hold). We copy "
        "that: six short movies spread across one 50 Hz rotation of the motor, "
        "then keep the worst spectrum. That stops the answer from depending on "
        "whether we happened to catch a high-duty or low-duty slice of the cycle."
    ))

    story.append(P("Five signals you can look at", "h2"))
    story.append(P(
        "Beside the EMC calculation, the same switching movie is used to draw "
        "five traces a designer would recognise on an oscilloscope:"
    ))
    story.append(simple_table(
        ["Trace", "What you are looking at", "Why it is there"],
        [
            ["DC-link voltage", "The 565 V “tank” after the rectifier, with a 300 Hz wobble plus a 2×carrier ripple.", "Shows that the tank is not perfectly stiff."],
            ["Motor PWM voltage", "The slammed on/off voltage between two motor wires.", "The actual PWM the motor is fed."],
            ["Motor current", "That PWM, filtered by the motor’s coil (inductance). Current cannot jump, so the stairs become a sawtooth.", "Where ripple lives. Smoother at high carrier."],
            ["Common-mode voltage", "The three-leg average. The villain of conducted EMC.", "The cause, not the symptom."],
            ["Input current", "What the drive draws from the building, through a 6-pulse rectifier.", "Power quality on the mains side."],
        ],
        [CONTENT_W * 0.22, CONTENT_W * 0.48, CONTENT_W * 0.30],
    ))

    story.append(P("From time to frequency", "h2"))
    story.append(P(
        "A movie of voltage versus time is hard to judge against a radio rule. "
        "Laboratories therefore look at <b>frequency</b>: how much energy sits "
        "at each pitch. A Fourier transform (FFT) is the mathematical ear that "
        "hears a messy slam as a stack of tones. We then pretend to be a "
        "CISPR receiver with a 9 kHz listening window, because that is how "
        "conducted-emission tests are actually done."
    ))
    story.append(P(
        "The result is a curve: emission versus frequency, in dBuV (decibels "
        "above one microvolt). Higher on that plot means louder electrical "
        "noise at that pitch."
    ))

    story.append(P("Three neighbourhoods: Band A, B, C", "h2"))
    story.append(P(
        "We do not stare at thousands of frequencies equally. We split the "
        "conducted-emission range into three neighbourhoods that behave "
        "differently, then worry most about the worst one."
    ))
    story.append(png_image(figure_bands(), CONTENT_W))
    story.append(P(
        "Figure. The assumed limit line, and the three bands. This line is a "
        "stand-in, not the copyrighted legal table.",
        "caption",
    ))
    story.append(simple_table(
        ["Band", "Frequencies", "Personality in this tool"],
        [
            ["A", "150 kHz – 500 kHz", "Closest to the switching harmonics themselves. Carrier rate shows up here strongly."],
            ["B", "0.5 – 5 MHz", "The middle. Cable length and shielding both matter."],
            ["C", "5 – 30 MHz", "The high hiss. Steep dv/dt and cable ringing live here."],
        ],
        [CONTENT_W * 0.12, CONTENT_W * 0.28, CONTENT_W * 0.60],
    ))

    story.append(P("The limit line, honestly", "h2"))
    story.append(P(
        "A laboratory compares your product to a legal curve and says pass or "
        "fail. The real elevator-sector table (EN 12016) is copyrighted, so we "
        "are not allowed to reprint it. Instead we drew a <b>synthetic</b> "
        "cousin: 79 dBuV at 150 kHz, sloping to 73 dBuV at 500 kHz, flat to "
        "5 MHz, then down to 70 dBuV at 30 MHz. It is anchored on publicly "
        "described CISPR 11 Group 1 Class A quasi-peak levels. Sitting under "
        "this cousin is <b>not</b> a claim about EN 12016."
    ))
    story.append(callout(
        "MARGIN",
        "Margin is the gap between your simulated noise and that assumed line, "
        "in decibels. Positive margin means “we are under the line at this "
        "frequency.” Negative means “we popped over.” Zero means sitting "
        "exactly on it. The tool’s whole personality is: talk in margins and "
        "risk, never in certificates.",
        "note",
    ))

    story.append(P("Power quality is a different question", "h2"))
    story.append(P(
        "<b>THD</b> (total harmonic distortion) asks: how much extra junk is "
        "riding on the 50 Hz current, as a percentage? That is a "
        "<b>power-quality</b> question about heat, flicker, and the building "
        "supply. It is <b>not</b> an EMC-limit question. The app shows THD on "
        "motor current and input current in a separate panel so the two "
        "subjects cannot be accidentally mashed into one score. The input "
        "filter knob moves THD. It is not an input to the EMC models, on "
        "purpose."
    ))

    # ------------------------------------------------------------------ part 4
    story.append(P("PART 4", "part"))
    story.append(P("The score, and the machine-learning helper", "h1"))
    story.append(P("From a gap in dB to a number you can talk about", "h2"))
    story.append(P(
        "Three margins (one per band) are turned into a 0–100 <b>risk score</b>. "
        "The mapping is a smooth S-shape: sitting exactly on the limit scores "
        "50. Being comfortably below scores toward 100. Being clearly over "
        "scores toward 0. The overall score is half the average of the three "
        "bands and half the worst band, so one terrible band cannot hide "
        "behind two pretty ones."
    ))
    story.append(png_image(figure_score(), CONTENT_W))
    story.append(P(
        "Figure. How a margin becomes a band score. Green dashed line: LOW "
        "tier. Red dashed line: HIGH tier. Higher score means more simulated "
        "headroom, not a laboratory pass.",
        "caption",
    ))
    story.append(simple_table(
        ["Score", "Tier", "How to say it out loud"],
        [
            ["70–100", "LOW risk", "The simulated noise sits comfortably under the assumed line in every band."],
            ["40–69", "MODERATE risk", "At least one band is close, or the overall headroom is thin."],
            ["0–39", "HIGH risk", "One or more bands are predicted to exceed the assumed line by a material amount."],
        ],
        [CONTENT_W * 0.16, CONTENT_W * 0.22, CONTENT_W * 0.62],
    ))
    story.append(P(
        "The ± number next to the score is <b>not</b> “how wrong we are versus "
        "a real chamber.” It is how much a small family of similar models "
        "disagree with each other when they are all trained on the same "
        "simulator. That is a floor on true uncertainty, not an estimate of it.",
        "muted",
    ))

    story.append(P("Why use machine learning at all?", "h2"))
    story.append(P(
        "The simulator already gives an answer. The models sit on top of it "
        "for three reasons that are easy to say without maths:"
    ))
    story.append(bullets([
        "<b>A calibrated guess of the margin</b> in each band, with a known error versus the simulator (not versus hardware).",
        "<b>A promise about direction.</b> Each input is told “this knob is only allowed to make risk worse, or only allowed to make it better.” The trees cannot learn that adding shielding increases emissions. That promise is called a <b>monotone constraint</b>.",
        "<b>An explanation.</b> For the current design, the tool can say which knobs pushed the margin up or down, in decibels.",
    ]))
    story.append(P(
        "The models are gradient-boosted trees (XGBoost): many small decision "
        "trees added together. They were trained on thousands of simulated "
        "designs, not on chamber measurements. Eighteen numbers go in: the six "
        "design knobs, plus four spectral descriptors per band. What comes out "
        "per band is a margin in dB (and a probability of sitting over the "
        "line, which we treat as supporting colour, not as a verdict)."
    ))

    story.append(P("“Why this margin” — SHAP", "h2"))
    story.append(P(
        "SHAP is a way of splitting a prediction into a pile of signed "
        "contributions that add back up to the answer. Think of a restaurant "
        "bill: the total is real, and SHAP is a fair split of who ordered "
        "what. For us the “bill” is the predicted margin of the worst band, "
        "in decibels."
    ))
    story.append(P(
        "We use the tree’s own exact split (not a slow approximation). Green "
        "bars are knobs that gave you headroom. Red bars are knobs that took "
        "it away. The bars are sorted by how big they are, largest at the "
        "top, so you read the strongest story first."
    ))
    story.append(P(
        "A quiet but important detail: this particular waterfall is computed "
        "from a <b>design-only</b> model that sees only the six knobs, not "
        "the spectrum. That is why it can talk about cable and shielding as "
        "causes, instead of merely pointing at a spectral peak that is itself "
        "a symptom. The identity check “all the bars plus the starting bias "
        "equals the predicted margin” is measured and shown on the Model "
        "Validation page. Typical error is about 8 millionths of a decibel."
    ))

    story.append(P("“What to change next”", "h2"))
    story.append(P(
        "This list is not the same ranking as SHAP, and that is deliberate. "
        "SHAP explains why you are where you are, including knobs you already "
        "maxed out (a great shield still “contributes”). The recommendations "
        "only suggest knobs that still have room to move, ranked by remaining "
        "headroom. If the two lists disagree, read the sentence under them: "
        "they are answering different questions."
    ))
    story.append(P(
        "One of those recommendations is usually: drop the carrier toward "
        "about 4 kHz. That lowers every band at once, because fewer slams "
        "per second means fewer spectral lines in the measurement range. It "
        "is also not free — which is the next chapter."
    ))

    story.append(P("How we check the models are not lying about physics", "h2"))
    story.append(P(
        "Two homework tests are run on the trained files and shown in the app:"
    ))
    story.append(bullets([
        "<b>Monotonicity.</b> Two hundred random designs. Each of the six constrained knobs is swept while the others stay put. The risk score is only allowed to move in the physically legal direction. The committed suite reports 100% consistent.",
        "<b>SHAP additivity.</b> For many designs, add up the waterfall plus the bias. It must recover the predicted margin. Mean error about 7.8 x 10<sup>-6</sup> dB.",
    ]))
    story.append(P(
        "These tests prove the models obey their own rules. They do <b>not</b> "
        "prove the models match a chamber. That would need measured hardware, "
        "which this project does not have yet.",
        "muted",
    ))

    # ------------------------------------------------------------------ part 5
    story.append(P("PART 5", "part"))
    story.append(P("Nothing is free: the design trade-offs", "h1"))
    story.append(P(
        "If lowering the switching frequency always helped EMC and hurt "
        "nothing else, every drive would run at 3 kHz and this chapter would "
        "be empty. Two other physical facts get worse as the carrier falls. "
        "They are the same two costs the “what to change next” text already "
        "names: <b>audible noise</b> and <b>motor current ripple</b>."
    ))
    story.append(P("Audible noise (acoustic risk)", "h2"))
    story.append(P(
        "Human ears, roughly, hear 20 Hz to 20 kHz. A PWM carrier is a tone "
        "(plus sidebands) at the switching frequency. At 3–8 kHz many people "
        "hear a whine. At 16 kHz most adults do not. The tool does not "
        "simulate a cabin, a speaker, or loudness in phons. It uses a simple "
        "S-shaped curve: high risk at 3 kHz, about 50 at 8 kHz, low at 16 kHz. "
        "That is a <b>proxy</b> — a stand-in number — labelled as such on the "
        "chart."
    ))
    story.append(P("Motor current ripple (ripple cost)", "h2"))
    story.append(P(
        "The motor is a coil of wire. A coil hates sudden changes of current; "
        "it prefers to ramp. Between PWM slams the current drifts up and down. "
        "That wobble is <b>ripple</b>. The standard textbook envelope, for a "
        "two-level inverter at the worst duty cycle, is:"
    ))
    story.append(P(
        "dI ~ V<sub>dc</sub> / (4 x L<sub>motor</sub> x f<sub>sw</sub>)",
        "h2",
    ))
    story.append(P(
        "In words: ripple current gets bigger if the tank voltage is bigger, "
        "if the motor coil is smaller, or if you switch less often. We do not "
        "ask the user for motor inductance; we reuse the same 6 millihenry "
        "assumption the rest of the simulator already uses, and the same 565 V "
        "DC link. The 0–100 “ripple cost” is that dI, gently squashed with a "
        "curve (called tanh) so huge currents still land near 100 and tiny ones "
        "near 0, sharing a scale with the other scores. It is not a torque "
        "simulation and not a heat calculation."
    ))
    story.append(png_image(figure_tradeoff(), CONTENT_W))
    story.append(P(
        "Figure. Both named costs fall as switching frequency rises. EMC "
        "headroom does the opposite (not drawn here: it is the model’s job). "
        "That opposition is the trade-off.",
        "caption",
    ))
    story.append(P("Pareto: the set of designs that are not strictly worse", "h2"))
    story.append(P(
        "A point on the trade-off chart is <b>Pareto-optimal</b> if no other "
        "point in the sweep is better on EMC headroom <b>and</b> cheaper on "
        "the cost axis at the same time. The others are <b>dominated</b>: you "
        "would never pick them if you only cared about those two numbers."
    ))
    story.append(P(
        "Because acoustic risk and ripple cost are both driven by the same "
        "knob (switching frequency), they light up the <b>same</b> set of "
        "Pareto points. That is not a duplicate chart by accident. It is a "
        "finding: lowering the carrier to help EMC consistently costs both "
        "quieter operation and smoother current, together. Clicking a point "
        "loads that carrier into the main assessment and re-runs the full "
        "prediction — not a static tooltip."
    ))
    story.append(callout(
        "WHAT WE DELIBERATELY LEFT OUT",
        "Switching loss (heat in the transistors) also changes with frequency, "
        "but it moves the <b>same</b> way as EMC: both get better when you "
        "slow the carrier. Charting it as a “trade-off” would have been a lie. "
        "It was removed. Only costs that actually oppose EMC along this sweep "
        "are drawn.",
        "honesty",
    ))

    # ------------------------------------------------------------------ part 6
    story.append(P("PART 6", "part"))
    story.append(P("What you see in the app, in order", "h1"))
    story.append(simple_table(
        ["Step", "Screen", "What to notice"],
        [
            ["1", "Select a drive", "Six starting personalities (gearless passenger, ultra-rise, machine-room-less, freight, silicon-carbide high-speed, escalator) or a blank custom."],
            ["2", "Set parameters", "The knobs, with the legal ranges. Switching frequency is 3–16 kHz on purpose: that is real VFD practice, not a random slider."],
            ["3", "Review risk", "The framing sentence first (this is a simulation). Then the LOW / MODERATE / HIGH badge, the score with ±, and the spectrum against the assumed line."],
            ["", "Band table", "The three margins. Negative is a simulated exceedance of the assumed curve, not a legal fail."],
            ["", "Signals and THD", "The five oscilloscope traces, and power-quality THD kept separate from EMC."],
            ["", "Why this margin", "SHAP waterfall in dB, largest |contribution| at the top."],
            ["", "What to change next", "Up to three levers with room left. May disagree with SHAP; that is explained on the page."],
            ["", "Design trade-offs", "Two charts: EMC vs acoustic, EMC vs ripple. Same Pareto set. Click a point to re-assess."],
            ["", "Compare / PDF", "A second configuration beside the first, and a “Virtual EMC Pre-Compliance Report” that recomputes from parameters (you cannot fake the numbers in the PDF)."],
            ["", "Methodology / Validation", "The assumptions, and the measured monotonicity and SHAP-identity headlines."],
        ],
        [CONTENT_W * 0.10, CONTENT_W * 0.24, CONTENT_W * 0.66],
    ))

    # ------------------------------------------------------------------ part 7
    story.append(P("PART 7", "part"))
    story.append(P("What this is not (please read this out loud)", "h1"))
    story.append(callout(
        "THE TOOL DOES NOT CERTIFY ANYTHING",
        "It does not predict EN 12016 outcomes. It does not replace an "
        "accredited laboratory. A LOW badge is not a pass. A HIGH badge is "
        "not a legal fail. Both are design-risk flags against a synthetic "
        "limit, inside a physics simulation.",
        "honesty",
    ))
    story.append(bullets([
        "<b>No measured hardware</b> is in the training loop. Every accuracy number is “how well the model copies our own simulator,” never “how well either matches a chamber.”",
        "<b>The limit line is a stand-in.</b> Direction and size of a design change are the reliable output. An absolute tier is only as good as that curve.",
        "<b>Lumped models.</b> Cable, shield and rectifier are first-order. They get the direction of change right. They do not know your cabinet’s private parasitics.",
        "<b>Radiated emissions, immunity, and functional safety</b> are out of scope.",
        "<b>Acoustic and ripple numbers are proxies</b>, not microphone measurements and not a finite-element motor model.",
        "<b>calibrate.py</b> can absorb real chamber leftovers when a CSV exists. Until then, every figure stays relative to the simulator.",
    ]))
    story.append(P(
        "If someone asks “will this pass the lab?” the only honest sentence is: "
        "<i>we do not know; here is how the simulated risk moved when you "
        "changed the design, and here is why.</i>"
    ))

    # ------------------------------------------------------------------ glossary
    story.append(P("PART 8", "part"))
    story.append(P("Pocket glossary", "h1"))
    story.append(P(
        "A last pass through every specialised word in the product, in one "
        "place. If you can explain each line to a curious twelve-year-old, "
        "you have understood the build."
    ))
    glossary = [
        ("Carrier / switching frequency", "The drumbeat of the inverter, in kHz. How often the electronic taps slam."),
        ("CISPR / EN 12016", "Families of EMC rules. CISPR is a general radio-disturbance method. EN 12016 is the elevator-sector standard. We do not reprint EN 12016."),
        ("Common mode", "The part of the voltage that all three motor wires share. It pushes noise into earth. Opposite of differential, which is the useful voltage between wires."),
        ("Conducted emissions", "Noise that leaves through the cables. What this tool estimates. Opposite of radiated (through the air)."),
        ("dB, dBuV", "A compressed comparison scale. dBuV is “decibels above one microvolt,” the usual unit on a conducted-emission plot."),
        ("Differential", "The useful voltage between two motor phases. Not the main EMC villain."),
        ("dv/dt", "Steepness of each voltage slam, in volts per microsecond."),
        ("Dwell / max-hold", "Listen at one frequency for a while and remember the loudest moment. What a real receiver does; what our six segments copy."),
        ("EMC", "Electromagnetic compatibility: machines that do not ruin each other’s day."),
        ("Emissions / immunity", "Weather you send versus weather you can survive. This tool: emissions only."),
        ("FFT", "A mathematical ear that turns a time movie into a stack of tones."),
        ("Harmonic", "A higher extra tone at a multiple of a base frequency. PWM is full of them."),
        ("Inductance (L)", "A coil’s stubbornness against changing current. Why motor current has ripple instead of copying PWM instantly."),
        ("Inverter / VFD", "The box that rebuilds wall electricity into a controllable motor supply by switching."),
        ("LISN", "A standard box that gives the test instrument a known view of the noise on the power line."),
        ("Margin", "How far under (or over) the assumed limit you are, in dB."),
        ("Monotone constraint", "A rule that forbids the ML model from learning a physically backwards story."),
        ("Parasitic capacitance", "An accidental capacitor formed by cable metal sitting near earth."),
        ("Pareto-optimal", "Not strictly worse than another option on both goals at once."),
        ("PWM", "Pulse-width modulation: encoding a smooth average in the widths of hard on/off pulses."),
        ("Quasi-peak (QP)", "A receiver weighting used in emission tests. Our limit anchors are QP-style numbers."),
        ("RBW", "Resolution bandwidth: how wide a slice of frequency the receiver listens to at once. Here, 9 kHz."),
        ("Rectifier", "The front end that turns AC from the wall into DC for the tank."),
        ("Resonance", "A cable ringing at a pitch set by its length, like a pipe."),
        ("Ripple", "The leftover wobble on motor current between PWM slams."),
        ("Risk score", "0–100 summary of simulated headroom. Higher is more headroom, not a pass."),
        ("SHAP", "A fair split of a prediction into per-knob contributions that add back up."),
        ("Shield / screen", "The metal braid and filters that try to keep common-mode current inside."),
        ("Simulation-consistency", "How well the ML copies the simulator — not how well either copies hardware."),
        ("Synthetic limit", "Our stand-in emission curve, not the copyrighted legal table."),
        ("THD", "Total harmonic distortion: extra junk on a 50 Hz current, as a percentage. Power quality, not EMC."),
        ("Trapezoid", "A pulse with sloping sides. The slope is dv/dt. The spectrum of that shape is a classic EMI textbook result."),
    ]
    story.append(simple_table(
        ["Word", "In one breath"],
        [[a, b] for a, b in glossary],
        [CONTENT_W * 0.28, CONTENT_W * 0.72],
    ))

    story.append(Spacer(1, 8))
    story.append(HairRule(CONTENT_W, RULE_STRONG, 0.8))
    story.append(Spacer(1, 6))
    story.append(P(
        "This document describes EMC Advisor as built: a physics simulator, "
        "monotone-constrained models, an explained risk score, and an honest "
        "trade-off between EMC headroom, audible noise, and motor current "
        "ripple. It is a map of the product. It is not a certificate for any "
        "elevator, anywhere."
    ))
    story.append(P(
        "Guidance on framing was reviewed against publicly discussed "
        "elevator-drive EMC practice. That review is not a product endorsement "
        "and does not constitute validation against hardware.",
        "small",
    ))

    buffer = io.BytesIO()
    document = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN + 4 * mm,
        bottomMargin=PAGE_MARGIN,
        title="EMC Advisor explained from zero",
        author="EMC Advisor",
        subject="Plain-language documentation of the Virtual EMC Pre-Compliance Advisor",
    )
    frame = Frame(
        PAGE_MARGIN, PAGE_MARGIN, CONTENT_W,
        PAGE_H - 2 * PAGE_MARGIN - 4 * mm,
        id="body",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])
    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf = build()
    OUT_PATH.write_bytes(pdf)
    print(f"Wrote {OUT_PATH} ({len(pdf)} bytes)")


if __name__ == "__main__":
    sys.exit(main() or 0)
