"""
Build a company-ready study briefing for EMC Advisor.

Run from the backend folder:

    .venv\\Scripts\\python.exe tools\\company_briefing_pdf.py

Writes ``docs/EMC_Advisor_Company_Briefing.pdf`` at the repository root.
"""

from __future__ import annotations

import io
import math
import sys
from datetime import date
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")

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
OUT_PATH = ROOT / "docs" / "EMC_Advisor_Company_Briefing.pdf"

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
        leading=leading if leading is not None else size * 1.42,
        textColor=colour,
        spaceAfter=space_after,
        spaceBefore=space_before,
        alignment=TA_LEFT,
        charSpace=tracking,
    )


STYLES = {
    "kicker": _style("kicker", 8, font=BOLD_FONT, colour=INK_MUTED, tracking=1.4),
    "cover_title": _style("cover_title", 28, leading=32, font=BOLD_FONT, space_before=4),
    "cover_sub": _style("cover_sub", 11.5, leading=16, colour=INK_MUTED, space_before=8),
    "part": _style(
        "part", 8, font=BOLD_FONT, colour=INK_MUTED, tracking=1.4,
        space_before=14, space_after=3,
    ),
    "h1": _style("h1", 14.5, leading=18, font=BOLD_FONT, space_before=2, space_after=6),
    "h2": _style("h2", 11, leading=14, font=BOLD_FONT, space_before=10, space_after=4),
    "body": _style("body", 9.3, leading=13.4, space_after=7),
    "body_tight": _style("body_tight", 9.3, leading=13.4, space_after=3),
    "muted": _style("muted", 9.3, leading=13.4, colour=INK_MUTED, space_after=7),
    "small": _style("small", 7.6, leading=10.8, colour=INK_MUTED, space_after=4),
    "caption": _style("caption", 7.4, leading=10.2, colour=INK_FAINT, space_after=10, space_before=2),
    "th": _style("th", 7.4, leading=10, font=BOLD_FONT, colour=INK_FAINT),
    "td": _style("td", 8.2, leading=11.2),
    "td_muted": _style("td_muted", 8.2, leading=11.2, colour=INK_MUTED),
    "box_title": _style("box_title", 8, font=BOLD_FONT, tracking=0.6, space_after=3),
    "box_body": _style("box_body", 8.6, leading=12.2),
    "footer": _style("footer", 7, colour=INK_FAINT),
    "toc_item": _style("toc_item", 9.4, leading=14),
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
    bar = {
        "honesty": FAIL_COLOUR,
        "never": FAIL_COLOUR,
        "say": PASS_COLOUR,
        "note": INK,
    }[kind]
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
    wrapper = Table(
        [[inner]],
        colWidths=[CONTENT_W],
    )
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
    style_cmds: List[tuple] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE_STRONG),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
    ]
    table.setStyle(TableStyle(style_cmds))
    return table


def png_image(png: bytes, width: float) -> Image:
    from reportlab.lib.utils import ImageReader

    reader = ImageReader(io.BytesIO(png))
    iw, ih = reader.getSize()
    return Image(io.BytesIO(png), width=width, height=width * (ih / iw))


def figure_limit_curve() -> bytes:
    freqs = [0.15, 0.5, 5.0, 30.0]
    limits = [79.0, 73.0, 73.0, 70.0]
    fig = Figure(figsize=(7.2, 2.35), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(freqs, limits, color="#0A0A0A", linewidth=1.6)
    ax.scatter(freqs, limits, color="#0A0A0A", s=18, zorder=3)
    for x, y, label in zip(freqs, limits, ["79", "73", "73", "70"]):
        ax.annotate(
            f"{label} dBµV",
            (x, y),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=7,
            color="#4A4A4A",
        )
    ax.set_xscale("log")
    ax.set_xlim(0.12, 35)
    ax.set_ylim(66, 84)
    ax.set_xlabel("Frequency (MHz)", fontsize=7.5, color="#4A4A4A")
    ax.set_ylabel("Assumed QP limit (dBµV)", fontsize=7.5, color="#4A4A4A")
    ax.tick_params(labelsize=7, colors="#4A4A4A")
    ax.axvspan(0.15, 0.5, color="#0A0A0A", alpha=0.04)
    ax.axvspan(0.5, 5, color="#0A0A0A", alpha=0.02)
    ax.axvspan(5, 30, color="#0A0A0A", alpha=0.05)
    ax.text(0.27, 67.2, "Band A", fontsize=7, color="#6B6B6B", ha="center")
    ax.text(1.6, 67.2, "Band B", fontsize=7, color="#6B6B6B", ha="center")
    ax.text(12, 67.2, "Band C", fontsize=7, color="#6B6B6B", ha="center")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.35)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor="white")
    return buf.getvalue()


def figure_score_map() -> bytes:
    margins = [i / 10.0 for i in range(-160, 161)]
    scores = [max(0.0, min(100.0, 50.0 + 50.0 * math.tanh(m / 8.0))) for m in margins]
    fig = Figure(figsize=(7.2, 2.35), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(margins, scores, color="#0A0A0A", linewidth=1.6)
    ax.axhline(70, color="#1A7F45", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.axhline(40, color="#B4232A", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.axvline(0, color="#C9C9C9", linewidth=0.7)
    ax.text(12.5, 73.5, "LOW  ≥ 70", fontsize=7, color="#1A7F45")
    ax.text(12.5, 32.5, "HIGH  < 40", fontsize=7, color="#B4232A")
    ax.set_xlim(-16, 16)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Predicted margin to the assumed limit (dB)", fontsize=7.5, color="#4A4A4A")
    ax.set_ylabel("Band score", fontsize=7.5, color="#4A4A4A")
    ax.tick_params(labelsize=7, colors="#4A4A4A")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.35)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor="white")
    return buf.getvalue()


def on_page(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFillColor(INK_FAINT)
    canvas.setFont(BODY_FONT, 7)
    if doc.page > 1:
        canvas.drawString(PAGE_MARGIN, PAGE_H - 12 * mm, "EMC Advisor  ·  company briefing")
        canvas.drawRightString(PAGE_W - PAGE_MARGIN, PAGE_H - 12 * mm, "Not a certification document")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(PAGE_MARGIN, PAGE_H - 13.2 * mm, PAGE_W - PAGE_MARGIN, PAGE_H - 13.2 * mm)
        canvas.line(PAGE_MARGIN, 12.5 * mm, PAGE_W - PAGE_MARGIN, 12.5 * mm)
        canvas.drawString(
            PAGE_MARGIN,
            8.5 * mm,
            "Simulation-based risk indicator. Does not predict EN 12016 outcomes.",
        )
        canvas.drawRightString(PAGE_W - PAGE_MARGIN, 8.5 * mm, str(doc.page))
    canvas.restoreState()


def build() -> bytes:
    story: List[Flowable] = []

    # ------------------------------------------------------------------ cover
    story.append(Spacer(1, 8 * mm))
    story.append(P("EMC ADVISOR  ·  STUDY BRIEFING", "kicker"))
    story.append(HairRule(CONTENT_W, INK, 1.1))
    story.append(P("Explain this product<br/>to a real company.", "cover_title"))
    story.append(P(
        "A complete briefing on the Virtual EMC Pre-Compliance Advisor: "
        "what electromagnetic compatibility is, why elevator drives emit noise, "
        "how the physics simulator and the machine-learning models work, "
        "what every number on the screen actually means, and how to present "
        "the tool without overselling it.",
        "cover_sub",
    ))
    story.append(Spacer(1, 8))
    story.append(P(
        f"Internal study document  ·  {date.today().isoformat()}  ·  "
        "artifact version 2.0.0  ·  written for a reader who does not already "
        "know EMC, power electronics, or machine learning.",
        "small",
    ))
    story.append(Spacer(1, 10))
    story.append(callout(
        "READ THIS BEFORE YOU WALK INTO THE ROOM",
        "This tool estimates conducted-EMC <b>risk from a physics simulation</b>. "
        "It does not predict EN 12016 certification. It has never been fitted to "
        "chamber measurements. The limit curve is a documented assumption, not "
        "the copyrighted standard table. Direction and size of a design change "
        "are the reliable output. An absolute LOW / MODERATE / HIGH badge is "
        "only as good as that assumed curve. If you remember one sentence, "
        "remember that one.",
        "honesty",
    ))

    story.append(P("How to use this document", "h2"))
    story.append(bullets([
        "<b>Ten minutes before a meeting:</b> memorise the one-minute pitch on the next page, then skip to Part V (what to say, what never to say, likely questions).",
        "<b>One evening of study:</b> read Parts I, III and V in order. That is enough to demo the product and survive a mixed audience of managers and engineers.",
        "<b>If an EMC engineer will be in the room:</b> also read Parts II and IV. They will ask about the limit line, common-mode current, monotone constraints, and the 99% accuracy trap.",
        "<b>If you only need a leave-behind:</b> print this PDF. Do not print a LOW-risk assessment PDF and call it a certificate.",
    ]))

    story.append(P("Contents", "h2"))
    story.append(P(
        "Part I &nbsp;&nbsp;&nbsp; The situation — EMC, labs, and why this exists<br/>"
        "Part II &nbsp;&nbsp; The physics — from a switching transistor to a spectrum<br/>"
        "Part III &nbsp; The product — what the software actually does<br/>"
        "Part IV &nbsp; The models — machine learning without the mystique<br/>"
        "Part V &nbsp;&nbsp; The meeting — pitch, demo, questions, landmines<br/>"
        "Part VI &nbsp; Reference — numbers you can quote, and a glossary",
        "toc_item",
    ))

    # -------------------------------------------------------------- one minute
    story.append(PageBreak())
    story.append(P("THE ONE-MINUTE PITCH", "part"))
    story.append(P("Memorise this. Then sit down.", "h1"))
    story.append(P(
        "We built a virtual pre-compliance advisor for conducted emissions on "
        "elevator variable-frequency drives. You give it six design numbers — "
        "switching frequency, edge speed, motor-cable length, shield quality, "
        "load current, and PWM strategy. It synthesises the inverter waveform, "
        "the common-mode current on the motor cable, and the spectrum a "
        "CISPR-style receiver would see. A physics-constrained model then "
        "returns a three-tier risk flag, an explanation of <i>why this design "
        "sits where it does</i>, and a separate list of <i>what still has room "
        "to change</i>.",
    ))
    story.append(P(
        "It is not a lab. It is not a certificate. The limit line is a synthetic "
        "CISPR-shaped envelope, because we cannot redistribute the EN 12016 "
        "tables. Every accuracy figure is agreement with our own simulator, "
        "not with hardware. What it <b>is</b> good for is the question that "
        "normally waits for a chamber booking: is this cable run, this carrier, "
        "and this shield going to be a problem — and which knob still moves "
        "the answer.",
    ))
    story.append(callout(
        "SAY THIS",
        "“Treat this as an early-design risk indicator. Use it to compare options "
        "and to walk into the lab less blind. Accredited measurement remains "
        "the only certification path.”",
        "say",
    ))
    story.append(callout(
        "NEVER SAY THIS",
        "“It certifies the drive.” “It predicts EN 12016.” “It is 99% accurate.” "
        "“The ensemble plus/minus is lab uncertainty.” “You can skip the chamber.” "
        "Any of those sentences ends the meeting.",
        "never",
    ))

    # ============================================================== PART I
    story.append(P("PART I  ·  THE SITUATION", "part"))
    story.append(P("What problem this is even for", "h1"))
    story.append(P(
        "An elevator drive is a box of power electronics. It takes building "
        "mains and turns it into a controlled waveform that runs a traction "
        "machine. To do that, transistors switch hundreds of volts on and off "
        "thousands of times per second. Every switch is a tiny radio "
        "transmitter. The motor cable in the shaft is the antenna. Other "
        "equipment in the building — and the lift’s own safety electronics — "
        "are the victims.",
    ))
    story.append(P(
        "<b>EMC</b> means electromagnetic compatibility: the drive must not "
        "pollute the electrical environment enough to upset neighbours, and it "
        "must keep working when the environment is noisy. For lifts and "
        "escalators in Europe the product-family standard people name in this "
        "context is <b>EN 12016</b> (immunity) together with the emission "
        "limits the sector actually tests against, which are CISPR / EN 55011 "
        "shaped. Getting that wrong late is expensive. A shielded cable, a "
        "common-mode choke, or a slower semiconductor can be a cheap line on "
        "a drawing and a painful retrofit on a site.",
    ))
    story.append(P(
        "The honest bottleneck is the <b>EMC chamber</b>. A proper conducted-"
        "emission test needs a Line Impedance Stabilisation Network (LISN), "
        "an EMI receiver, and time on a controlled site. You do not get a "
        "booking every time a firmware engineer wants to try 12 kHz instead "
        "of 8 kHz. So teams either over-design (cost, heat, acoustics) or "
        "under-design (fail the lab, slip the programme).",
    ))
    story.append(P(
        "This tool sits in that gap. It answers, in seconds, a <i>relative</i> "
        "question: given this design, relative to a documented assumed limit, "
        "where is the risk, and which parameter is carrying it. It does not "
        "replace the chamber. It makes the chamber a confirmation step instead "
        "of a discovery step.",
    ))

    story.append(P("EMC in plain language", "h2"))
    story.append(P(
        "Electricity is supposed to stay in wires. In real life, fast voltage "
        "changes push current through parasitic capacitance — the invisible "
        "capacitor between a cable and the metal of the shaft, or between a "
        "transistor and its heatsink. That current has to come home somehow. "
        "On the way home it flows through earth, through the mains, and through "
        "anything else that happens to be in the loop. A test house measures "
        "that homeward current as a voltage across a standardised 50 ohm "
        "network, in decibels relative to one microvolt (dBµV).",
    ))
    story.append(P(
        "Two families of test exist, and this product only claims one of them:",
    ))
    story.append(simple_table(
        ["Family", "What is measured", "In this tool"],
        [
            [
                "Conducted emissions",
                "Noise riding on the power cable, typically 150 kHz–30 MHz",
                "Yes — this is the whole product",
            ],
            [
                "Radiated emissions",
                "Noise leaving as a radio wave, typically 30 MHz–1 GHz+",
                "Out of scope",
            ],
            [
                "Immunity",
                "Whether the drive still works when someone else’s noise hits it",
                "Out of scope",
            ],
            [
                "Power quality / THD",
                "Distortion of motor and input current at 50 Hz harmonics",
                "Shown beside the risk score, not mixed into it",
            ],
        ],
        [32 * mm, 78 * mm, 65 * mm],
    ))
    story.append(P(
        "Conducted and radiated are related — the same edges cause both — but "
        "a good conducted result is not a radiated pass. Do not let anyone "
        "collapse them in the meeting.",
        "caption",
    ))

    story.append(P("What a lab actually does", "h2"))
    story.append(P(
        "A conducted-emission test does not look at a pretty oscilloscope trace. "
        "An EMI receiver dwells on each frequency with a legally defined "
        "<b>resolution bandwidth</b> (9 kHz in CISPR band B, which is 150 kHz "
        "to 30 MHz). It max-holds what it sees, because the inverter’s duty "
        "cycle breathes over the 50 Hz output cycle and a single snapshot "
        "would lie. The result is a curve of dBµV versus frequency, compared "
        "against a limit line.",
    ))
    story.append(P(
        "Three sub-bands are how drive people usually talk about that curve:",
    ))
    story.append(simple_table(
        ["Band", "Range", "What tends to live there"],
        [
            ["A", "150–500 kHz", "Switching harmonics, the bulk of the envelope"],
            ["B", "500 kHz–5 MHz", "Still switching structure; often the tightest band here"],
            ["C", "5–30 MHz", "Edge speed, cable resonances, shield quality"],
        ],
        [22 * mm, 40 * mm, 113 * mm],
    ))
    story.append(Spacer(1, 6))

    story.append(P("Pre-compliance is not certification", "h2"))
    story.append(P(
        "<b>Certification</b> is an accredited laboratory measuring a physical "
        "product against a named standard, with a report that a notified body "
        "or a customer will accept. <b>Pre-compliance</b> is everything you do "
        "before that so you are not paying for surprises. Near-field probes, "
        "a LISN on a bench, a simulator, a risk model — all pre-compliance. "
        "Useful. Not the same legal object.",
    ))
    story.append(P(
        "EMC Advisor is labelled, in the UI and on every generated PDF, as a "
        "<b>Virtual EMC Pre-Compliance Report</b>. The word “certificate” was "
        "deliberately not used. There is no accreditation mark, no signature "
        "block, and no statement of conformity. If a slide deck behind you "
        "says “AI certification for EN 12016”, take the slide down before "
        "you speak.",
    ))

    # ============================================================== PART II
    story.append(PageBreak())
    story.append(P("PART II  ·  THE PHYSICS", "part"))
    story.append(P("From a transistor to a spectrum", "h1"))
    story.append(P(
        "You do not need a power-electronics degree. You need six pictures in "
        "your head, because every knob on the screen is one of these pictures "
        "with a number attached.",
    ))

    story.append(P("1. A VFD is a fast switch, not a dimmer", "h2"))
    story.append(P(
        "A variable-frequency drive does not turn the motor voltage “partly "
        "on”. It slams the DC link (here modelled as 565 V, a rectified 400 V "
        "three-phase supply) onto each motor phase, then slams it off again, "
        "thousands of times per second. The average of those pulses is the "
        "smooth voltage the motor wanted. The pulses themselves are the EMC "
        "problem. The pulse repetition rate is the <b>switching frequency</b> "
        "(3–16 kHz in this tool, the range of ordinary elevator IGBT and SiC "
        "carriers). Raise it and the motor is quieter and smoother. Raise it "
        "and you also put more switching edges into the 150 kHz–30 MHz "
        "measurement window, so the spectrum rises.",
    ))

    story.append(P("2. The edges, not the pulses, set the high-frequency noise", "h2"))
    story.append(P(
        "A perfect square pulse has energy that falls 20 dB per decade of "
        "frequency, then 40 dB per decade after a corner set by the rise time. "
        "That is the classical trapezoidal EMI model. Faster edges (higher "
        "<b>dv/dt</b>, in volts per microsecond) push that corner up. Silicon "
        "carbide devices are efficient partly because they switch very fast — "
        "and they are noisy in band C for the same reason. This tool lets "
        "dv/dt run from 500 to 10 000 V/µs. A 9.5 kV/µs SiC edge is a "
        "different EMC object from a 1.2 kV/µs freight-drive IGBT.",
    ))

    story.append(P("3. Common mode is the actual villain", "h2"))
    story.append(P(
        "The three inverter legs are called A, B and C. The voltage that drives "
        "the motor <i>between</i> those legs is differential and mostly stays "
        "in the motor. The voltage that drives current <i>into the earth</i> "
        "is the average of the three: <b>v<sub>cm</sub> = (v<sub>a</sub> + "
        "v<sub>b</sub> + v<sub>c</sub>) / 3</b>. For a two-level inverter that "
        "average is a staircase that steps six times per carrier period. That "
        "staircase is the conducted-emission source. This is why a prettier "
        "line-to-line PWM picture can still be a disaster on the LISN.",
    ))
    story.append(P(
        "Current then follows i = C · dv/dt. The C is parasitic capacitance "
        "from the motor cable to the shaft (modelled here as 100 pF per metre). "
        "Longer cable, larger C, more current. That is why an 85 m ultra-rise "
        "shaft cable is a different risk from a 12 m machine-room-less run, "
        "even if the inverter is “the same”.",
    ))

    story.append(P("4. The cable is also a radio resonator", "h2"))
    story.append(P(
        "A long cable is a transmission line. It rings at a quarter-wave "
        "resonance f ≈ v / (4 × length) and at the odd harmonics 3f, 5f. "
        "If you only modelled the first resonance, lengthening the cable would "
        "sometimes slide the peak <i>out</i> of the 5–30 MHz band and the "
        "model would stupidly report that longer cables are quieter there. "
        "The simulator therefore keeps three resonant modes, so more cable "
        "still means more coupling, which is the physics the machine-learning "
        "constraints later enforce.",
    ))

    story.append(P("5. A shield is an attenuator, not a magic zero", "h2"))
    story.append(P(
        "Shielding quality in the UI is a 0–1 slider that folds together cable "
        "screen, 360-degree termination, and any common-mode choke. Zero is "
        "unscreened. One is a well-terminated screen plus a choke, modelled as "
        "up to 26 dB of insertion loss and a high-frequency roll-off. Better "
        "shielding can only reduce the emission in this model. It cannot "
        "invent energy. That sign is later locked into the trees so a noisy "
        "corner of training data cannot learn the opposite.",
    ))

    story.append(P("6. PWM strategy changes how often the villain steps", "h2"))
    story.append(P(
        "Four modulation strategies are modelled. The number that enters the "
        "model is not a rank. It is a measured common-mode penalty in dB, "
        "relative to ordinary sinusoidal PWM, averaged in the simulator itself:",
    ))
    story.append(simple_table(
        ["Strategy", "Penalty vs SPWM", "Why"],
        [
            ["Sinusoidal PWM (SPWM)", "0 dB", "Reference. All three legs switch every carrier period."],
            ["Space-vector (SVPWM)", "+1.0 dB", "Same switch count, but zero-sequence injection adds common-mode on purpose (the usual DC-link utilisation trade-off)."],
            ["Randomised carrier", "+1.0 dB", "Does not reduce energy. Against a 9 kHz max-hold receiver at 3–16 kHz carriers, extra variance gives the detector more peaks to catch."],
            ["Discontinuous (DPWM)", "−2.5 dB", "Clamps one leg to a rail, removing about a third of commutations. Fewer impulses per second, lower measured lines."],
        ],
        [38 * mm, 32 * mm, 105 * mm],
    ))
    story.append(P(
        "Two limitations you should volunteer before you are asked. Real drives "
        "sometimes see DPWM worse at very low frequency because of extra "
        "zero-sequence content; that content lives around 150 Hz, three orders "
        "of magnitude below this measurement floor, so the model does not "
        "reproduce it. And randomised PWM is often sold as a 5–10 dB win; in "
        "<i>this</i> measurement it is slightly worse, because CISPR band B "
        "already cannot resolve the 3–16 kHz line structure you would be "
        "spreading. Randomisation starts to pay at much higher carriers "
        "(100 kHz-class SiC), which are outside this tool’s range.",
        "muted",
    ))

    story.append(P("How the simulator copies a receiver", "h2"))
    story.append(P(
        "A 328 microsecond snapshot at 200 million samples per second can see "
        "30 MHz, but it cannot see a 50 Hz motor cycle. If you analysed one "
        "snapshot you would be measuring luck: where in the fundamental you "
        "happened to land. A real receiver dwells. The simulator therefore "
        "takes <b>six short records</b> spaced across one 50 Hz period and "
        "max-holds their spectra, then integrates power across a 9 kHz "
        "resolution bandwidth. That is the curve on the dashboard.",
    ))
    story.append(P(
        "Power quality is a different timescale on purpose. Total harmonic "
        "distortion of motor and input current needs many 50 Hz cycles, so it "
        "is computed from a cheaper 200 kS/s record. Mixing those two windows "
        "would be a mistake: the EMC capture cannot resolve mains harmonics, "
        "and the PQ capture cannot see 30 MHz. The UI keeps them in separate "
        "panels for that reason. The line reactor / input-filter slider only "
        "moves THD. It is not an XGBoost feature, so it cannot pretend to "
        "fix motor-cable EMC.",
    ))

    # ============================================================== PART III
    story.append(P("PART III  ·  THE PRODUCT", "part"))
    story.append(P("What you are actually demoing", "h1"))
    story.append(P(
        "A three-step web app. Pick a starting profile (or a custom set of "
        "numbers). Adjust the knobs. Run an assessment. The backend is a "
        "Python API; the frontend is a React dashboard. One request simulates "
        "the physics, scores the design, explains it, and can print a "
        "matching PDF.",
    ))

    story.append(P("The pipeline, in one glance", "h2"))
    story.append(simple_table(
        ["Stage", "What happens", "What you can say"],
        [
            [
                "1. Parameters",
                "Six design numbers plus PWM type. Input filter is PQ-only.",
                "“These are drawing-office knobs, not lab traces.”",
            ],
            [
                "2. simulate.py",
                "Trapezoidal three-phase PWM, common-mode voltage, cable C and resonances, shield, LISN, plus four companion waveforms.",
                "“A lumped physics model, not SPICE and not a measurement.”",
            ],
            [
                "3. features.py",
                "Receiver-emulated FFT, three-band margins, motor/input THD.",
                "“Same shape of object a test house plots, against our curve.”",
            ],
            [
                "4. Models",
                "Per-band XGBoost classifier and margin regressor, monotone-constrained. Design-only models for SHAP.",
                "“Forbidden from learning anti-physics.”",
            ],
            [
                "5. Score",
                "Each band margin maps through tanh to 0–100. Overall is half the mean, half the worst band.",
                "“A design cannot hide a failing band behind two healthy ones.”",
            ],
            [
                "6. Explain",
                "SHAP waterfall, countermeasures, tornado, compare, history, PDF.",
                "“Why it sits here, versus what is still worth turning.”",
            ],
        ],
        [28 * mm, 78 * mm, 69 * mm],
    ))
    story.append(Spacer(1, 8))

    story.append(P("The six knobs", "h2"))
    story.append(simple_table(
        ["Knob", "Range", "If you raise it"],
        [
            ["Switching frequency", "3–16 kHz", "Risk up (more edges per second). Acoustics usually improve."],
            ["Switching dv/dt", "500–10 000 V/µs", "Risk up, especially 5–30 MHz. Efficiency usually improves."],
            ["Motor cable length", "1–120 m", "Risk up. Parasitic C and resonances both grow."],
            ["Shielding quality", "0–1", "Risk down. Screen + termination + CM choke."],
            ["Load current", "5–200 A", "Risk up, sub-linearly. Also raises input-current THD."],
            ["PWM strategy", "four types", "DPWM helps CM in this band; SVPWM and random slightly hurt."],
        ],
        [40 * mm, 38 * mm, 97 * mm],
    ))
    story.append(Spacer(1, 8))

    story.append(P("Starting profiles", "h2"))
    story.append(P(
        "The six cards on the first screen are representative starting points, "
        "not characterisations of anyone’s shipped product. Two of them are "
        "deliberately uncomfortable, so you can show a mitigation path without "
        "inventing a disaster live.",
    ))
    story.append(simple_table(
        ["Profile", "Why it exists in the demo"],
        [
            ["Standard Gearless VFD", "The volume baseline: 8 kHz, 30 m, 45 A. Often the “this looks fine” case."],
            ["High-Speed Ultra-Rise", "The teaching case. 85 m shaft cable. Cable length will dominate the explanation."],
            ["Compact MRL Controller", "Short cable, weak screen, 16 kHz for acoustics. Different failure mode."],
            ["Freight Heavy-Duty", "High current, slow edges, DPWM, basic screen."],
            ["SiC Regenerative", "Very fast edges offset by a serious shield. Band C conversation."],
            ["Escalator VVVF", "Short, well-routed, moderate carrier. The relaxed end of the set."],
        ],
        [48 * mm, 127 * mm],
    ))
    story.append(Spacer(1, 8))

    story.append(P("The assumed limit — the single most important caveat", "h2"))
    story.append(P(
        "EN 12016 tables are copyrighted. They are not in this repository and "
        "they are not in this PDF. The tool uses a synthetic envelope anchored "
        "on publicly described CISPR 11 Group 1 Class A quasi-peak levels "
        "(79 dBµV around 150 kHz, 73 dBµV above), with two tapers that are "
        "<b>this tool’s own assumption</b>, including a slight tightening to "
        "70 dBµV at 30 MHz. Sitting below this curve is not a statement about "
        "EN 12016 conformity.",
    ))
    story.append(png_image(figure_limit_curve(), CONTENT_W))
    story.append(P(
        "Figure. Synthetic conducted-emission envelope used as the scoring anchor. "
        "Bands A / B / C are the three ranges the models score separately.",
        "caption",
    ))
    story.append(callout(
        "IF THEY ONLY REMEMBER ONE CAVEAT",
        "Absolute LOW / MODERATE / HIGH is only as trustworthy as this curve. "
        "Relative movement — shorten the cable, slow the edge, improve the "
        "screen — is the output you can defend today. Swap the anchors when "
        "you have the right to use a named table; the rest of the pipeline stays.",
        "honesty",
    ))

    story.append(P("How a margin becomes a badge", "h2"))
    story.append(P(
        "Each band’s predicted margin (limit minus emission, in dB) is mapped "
        "to a 0–100 band score with a saturating function. Zero dB of margin "
        "is exactly 50. About +8 dB is a high score; about −8 dB is a low one. "
        "The overall risk score is half the average of the three band scores "
        "and half the worst of them, so a single bad band cannot be averaged "
        "away. Tiers: <b>LOW 70–100</b>, <b>MODERATE 40–69</b>, "
        "<b>HIGH 0–39</b>. Higher score means more simulated headroom, not a "
        "lab pass.",
    ))
    story.append(png_image(figure_score_map(), CONTENT_W))
    story.append(P(
        "Figure. Band score versus predicted margin. The green and red lines "
        "are the LOW and HIGH cut-overs on the overall 0–100 score.",
        "caption",
    ))

    story.append(P("How to read the results page", "h2"))
    story.append(simple_table(
        ["Panel", "Job", "Do not confuse it with"],
        [
            [
                "Risk badge + score ±",
                "Headline headroom against the assumed curve, plus ensemble spread.",
                "A certification verdict, or a lab uncertainty bar.",
            ],
            [
                "Per-band margins",
                "Which slice of 150 kHz–30 MHz is actually tight.",
                "The overall badge. Always look here before you talk.",
            ],
            [
                "Why this margin (SHAP)",
                "Exact attribution of the worst-band predicted margin, in dB, from the six design parameters. Largest |contribution| at the top. Red hurts margin, green helps.",
                "A to-do list. Shielding can be the second-biggest bar because it is already good.",
            ],
            [
                "What to change next",
                "Remaining headroom only. Ignores knobs already near a safe stop.",
                "The SHAP ranking. They are allowed to disagree. The UI says so.",
            ],
            [
                "Sensitivity tornado",
                "If I nudge this knob from here, how much does the score move.",
                "A global importance ranking for all designs.",
            ],
            [
                "Five signals + THD",
                "Physics you can look at: DC link, motor PWM, motor current, common-mode voltage, input current; plus motor/input THD.",
                "The EMC limit comparison. THD is power quality.",
            ],
            [
                "Compare + history",
                "Two configurations side by side; last three runs as a sparkline.",
                "A statistical study. It is a design conversation aid.",
            ],
        ],
        [38 * mm, 72 * mm, 65 * mm],
    ))
    story.append(Spacer(1, 8))
    story.append(P(
        "A concrete example from the ultra-rise teaching case, worst band "
        "500 kHz–5 MHz. The design-only margin model predicted about −12.3 dB. "
        "SHAP split that as: cable length about −18.3 dB, shielding about "
        "+10.5 dB (already helping), load current about −6.5 dB, then small "
        "negative terms from dv/dt, PWM and carrier. Countermeasures still "
        "put cable first, then pointed at dv/dt and carrier rather than at "
        "shielding — because shielding is already near the top of its range. "
        "That divergence is a feature. Explain it. Do not “fix” it in the "
        "meeting by claiming the two lists must match.",
    ))

    story.append(P("The PDF the app downloads", "h2"))
    story.append(P(
        "Same framing, same assumed limit, same disclaimer, printed in the "
        "same black-and-white language as the UI. Title: Virtual Pre-Compliance "
        "Assessment. Use it as a design note after a review. Do not staple it "
        "to a type-examination file.",
    ))

    # ============================================================== PART IV
    story.append(PageBreak())
    story.append(P("PART IV  ·  THE MODELS", "part"))
    story.append(P("Machine learning without the mystique", "h1"))
    story.append(P(
        "If you have never trained a model, this section is for you. If you "
        "have, skip to “the 99% accuracy trap” — that is the part people get "
        "wrong in front of customers.",
    ))

    story.append(P("What a model is, here", "h2"))
    story.append(P(
        "The physics simulator is the ground truth of this product. It already "
        "knows, for any design, what spectrum it would produce. Running it "
        "takes on the order of a couple of hundred milliseconds. That is fine "
        "for one assessment and painful for sweeping a design space or for "
        "interactive “what if I shorten the cable by 10 m” loops.",
    ))
    story.append(P(
        "So the project trained a <b>surrogate</b>: a function that looks at "
        "the same numbers and returns approximately the same margins, instantly. "
        "The family used is <b>XGBoost</b> — gradient-boosted decision trees. "
        "In English: it builds hundreds of small “if cable longer than X, "
        "and shield worse than Y, then add Z dB of trouble” rules, and adds "
        "them up. Trees are a good fit because the physics is full of "
        "thresholds (resonances moving between bands) and because trees can "
        "be forced to respect a direction.",
    ))
    story.append(P(
        "Per frequency band the tool fits two such functions: a classifier "
        "that estimates the probability the band is on the wrong side of the "
        "assumed limit, and a regressor that estimates the margin in dB. The "
        "headline score uses the margins. A third, smaller family of models "
        "sees only the six design parameters, with the spectrum hidden from "
        "them. Those <b>design-only</b> models are what the SHAP chart uses, "
        "so the explanation is about knobs a designer can turn, not about "
        "“the peak in band B was high” (which is just restating the answer).",
    ))

    story.append(P("Where the training data came from", "h2"))
    story.append(P(
        "Five thousand random designs were drawn inside the allowed ranges, "
        "simulated, and labelled with the three band margins. Split: 3 500 "
        "train / 750 validation / 750 test. There is <b>no hardware in that "
        "loop</b>. The models are students of the simulator. They can be "
        "brilliant students and still be wrong about a real shaft, because "
        "the teacher has never seen a real shaft.",
    ))

    story.append(P("The physics-informed part: monotone constraints", "h2"))
    story.append(P(
        "Ordinary machine learning will happily learn nonsense if the data "
        "wobbles. A noisy batch in which, by chance, a few long-cable designs "
        "looked quiet could teach the trees that cable length helps. That is "
        "unacceptable in a product you will show to drive engineers.",
    ))
    story.append(P(
        "XGBoost therefore receives, for every input, a hard sign. Increasing "
        "switching frequency, dv/dt, cable length, load current, PWM penalty, "
        "or any spectral “how hot is this band” feature may only <b>increase</b> "
        "failure risk. Increasing shielding quality may only <b>decrease</b> "
        "it. For the margin regressors the signs are flipped, because more "
        "margin is safer. The constraint is structural: every split in every "
        "tree is required to move the prediction in the legal direction. It "
        "is not a post-hoc check. After training, a sweep still verified that "
        "all six design-parameter constraints hold.",
    ))
    story.append(callout(
        "HOW TO SAY THIS TO A SCEPTICAL ENGINEER",
        "“The model is not allowed to conclude that a longer cable is quieter, "
        "or that a better shield is noisier, even if a pocket of simulated "
        "data suggests it. That is the difference between a curve-fit and a "
        "physics-informed surrogate.”",
        "say",
    ))

    story.append(P("The 99% accuracy trap", "h2"))
    story.append(P(
        "On held-out simulated designs, the full 18-feature models look almost "
        "perfect: balanced accuracy around 0.98–1.00, margin errors of a few "
        "tenths of a dB. That number is not a miracle and must not be on a "
        "slide titled “AI accuracy”. The models can see "
        "<b>band_*_peak_dbuv</b> — essentially the height of the same spectrum "
        "the label was computed from. Predicting the label from the peak is "
        "close to reading the answer out of the question paper.",
    ))
    story.append(P(
        "The honest figure is the <b>design-only ablation</b>: hide the "
        "spectrum, give the trees only the six knobs, and ask them to infer "
        "the margin. That is the job a designer actually has before anyone "
        "has measured anything. Those figures, on the same 750 test designs:",
    ))
    story.append(simple_table(
        ["Band", "Design-only balanced acc.", "Design-only margin MAE"],
        [
            ["150 kHz – 500 kHz", "0.964", "1.06 dB"],
            ["500 kHz – 5 MHz", "0.972", "1.23 dB"],
            ["5 MHz – 30 MHz", "0.939", "1.88 dB"],
        ],
        [50 * mm, 60 * mm, 65 * mm],
    ))
    story.append(P(
        "Still tight against the simulator — and still not lab accuracy. "
        "Band C is the hardest, which matches the physics: 5–30 MHz is where "
        "lumped cable and shield models are most likely to be wrong on a "
        "real fixture. If a customer later gives you chamber data, this is "
        "the band to watch.",
        "caption",
    ))
    story.append(P(
        "A related negative result is kept in the methodology on purpose. "
        "Re-simulating the same design with different noise seeds moves the "
        "margin by about 0.25 dB; the model tracks that rather than averaging "
        "it away. Do not claim the ML “filters simulator noise”. It does not.",
    ))

    story.append(P("SHAP, in one paragraph", "h2"))
    story.append(P(
        "SHAP (SHapley Additive exPlanations) is a way of splitting a "
        "prediction into a sum of per-feature contributions plus a baseline. "
        "For trees, XGBoost can compute this exactly (<i>pred_contribs</i>), "
        "not as a slow approximation. For the worst band, those contributions "
        "are in decibels of predicted margin and add back up to the "
        "prediction. Green / positive means “this knob, at this setting, is "
        "adding headroom”. Red / negative means it is taking headroom away. "
        "The chart is sorted by absolute value so you can see the large "
        "movers first, regardless of sign.",
    ))

    story.append(P("The ± band on the score", "h2"))
    story.append(P(
        "Five extra margin models were trained on bootstrap resamples of the "
        "same simulated data. The ± you see is their disagreement, mapped "
        "onto the 0–100 score. It is a floor on uncertainty, not lab "
        "uncertainty. All five members went to the same school. They cannot "
        "tell you what the school got wrong about reality.",
    ))

    story.append(P("The path to real measurements", "h2"))
    story.append(P(
        "The file <b>calibrate.py</b> already exists. It is built for tens of "
        "chamber points, not thousands. It learns a small residual — measured "
        "margin minus simulated margin — as a function of the six design "
        "parameters, with the same monotone signs, and clamps it so a "
        "correction cannot invert physics. It has been tested on synthetic "
        "CSVs. It has <b>not</b> been run on chamber data. Until it has, "
        "every figure remains relative to the simulator, and the product says "
        "so on screen.",
    ))
    story.append(P(
        "A few dozen real points would shift the absolute score toward lab "
        "reality, turn the ± into an empirical residual, and show which band "
        "the lumped cable model misplaces (likely 5–30 MHz). They would not "
        "replace the physics prior, and they would not turn this into a "
        "certification oracle.",
    ))

    # ============================================================== PART V
    story.append(PageBreak())
    story.append(P("PART V  ·  THE MEETING", "part"))
    story.append(P("How to present this without getting hurt", "h1"))

    story.append(P("Who is in the room", "h2"))
    story.append(simple_table(
        ["Person", "What they care about", "Lead with"],
        [
            [
                "Engineering manager",
                "Schedule, lab cost, whether this reduces surprises",
                "Pre-compliance gap. Demo ultra-rise versus a shorter cable. Do not open with XGBoost.",
            ],
            [
                "Drive / EMC engineer",
                "Whether the physics is adult, whether you overclaim",
                "Common-mode, CISPR RBW, limit-curve honesty, monotone constraints, design-only MAE.",
            ],
            [
                "Compliance / quality",
                "Whether anyone will wave this as a certificate",
                "The disclaimer, the PDF title, the missing signature block. Then the calibration hook.",
            ],
            [
                "Data / AI lead",
                "Whether this is a demo wrapped around a black box",
                "Synthetic labels, ablation versus full-model scores, SHAP identity, why spectral features are a cheat.",
            ],
            [
                "Executive",
                "What we would buy, fund, or partner on",
                "One-minute pitch. Then: chamber data is the unlock, not more UI.",
            ],
        ],
        [36 * mm, 62 * mm, 77 * mm],
    ))
    story.append(Spacer(1, 8))

    story.append(P("A twelve-minute demo script", "h2"))
    story.append(bullets([
        "<b>0:00 — Frame.</b> Say the one-minute pitch. Point at the disclaimer in the header. Do not skip this to “save time”.",
        "<b>1:30 — Pick Ultra-Rise.</b> “Eighty-five metres of motor cable in a tall shaft. This is the case that should look uncomfortable.”",
        "<b>2:30 — Parameters.</b> Flick dv/dt and shielding so they see they are ordinary design numbers. Mention the input-filter slider does not feed the EMC model.",
        "<b>4:00 — Results badge.</b> Read the framing sentence on the page out loud. Then the tier. Then the per-band margins. Never the badge alone.",
        "<b>6:00 — Why this margin.</b> Largest bar first. Cable will dominate. Shielding will likely be a large green bar. Say: “that is already working; it is why the recommendations may not tell you to add more shield.”",
        "<b>7:30 — What to change next.</b> Read the divergence sentence between the two panels. This is where trust is won or lost.",
        "<b>8:30 — Signals / THD.</b> Ten seconds on common-mode voltage versus motor PWM. “Different question, different timescale.”",
        "<b>9:30 — Compare.</b> Shorten the cable or drop dv/dt. Show the score move. Relative change is the product.",
        "<b>10:30 — Methodology.</b> Open it. Land on the limit-curve box and the design-only table. Close by asking for a chamber CSV, not for applause.",
    ]))

    story.append(P("Questions you will get", "h2"))
    story.append(P("<b>Does this certify the drive?</b>", "body_tight"))
    story.append(P(
        "No. Accredited measurement is the only certification path. This is "
        "a design-risk indicator against a synthetic envelope.",
    ))
    story.append(P("<b>Can we skip the chamber if the badge is LOW?</b>", "body_tight"))
    story.append(P(
        "No. A LOW result means the simulator is comfortable relative to our "
        "curve. It is a reason to walk into the lab calmer, not a reason to "
        "cancel the lab.",
    ))
    story.append(P("<b>What is the accuracy?</b>", "body_tight"))
    story.append(P(
        "Against our own physics, the design-only models are about 1.1–1.9 dB "
        "mean absolute error on margin, depending on the band. That is not "
        "accuracy against hardware, which is unmeasured. Do not quote 99% "
        "balanced accuracy from the full models. Those models can see the "
        "spectrum they are scored on.",
    ))
    story.append(P("<b>Why machine learning if you already simulate?</b>", "body_tight"))
    story.append(P(
        "Speed for sweeps, and a surface we can calibrate with sparse lab "
        "points without rewriting the simulator. The simulator remains the "
        "prior. We did not replace physics with a black box.",
    ))
    story.append(P("<b>Why not a full finite-element / SPICE model?</b>", "body_tight"))
    story.append(P(
        "Wrong job. Early design needs direction in seconds from a handful of "
        "drawing-office numbers. Lumped cable, first-order shield, and a "
        "trapezoidal inverter are enough for that, and they are honest about "
        "what they omit: fixture parasitics, connector details, cabinet "
        "layout, radiated paths.",
    ))
    story.append(P("<b>Is the limit EN 12016?</b>", "body_tight"))
    story.append(P(
        "No. EN 12016 is named because that is the lift-sector standard "
        "people will mention. The numerical table is not redistributable. Our "
        "curve is CISPR 11 Group 1 Class A shaped, with our own tapers, "
        "flagged everywhere as an assumption.",
    ))
    story.append(P("<b>Your SHAP list and your recommendations disagree.</b>", "body_tight"))
    story.append(P(
        "They should, sometimes. SHAP explains the current point, including "
        "factors already maxed out. Recommendations only suggest parameters "
        "with travel left. Shielding is the usual example.",
    ))
    story.append(P("<b>Random PWM is supposed to be better.</b>", "body_tight"))
    story.append(P(
        "Often, at high carriers where lines are resolved. In CISPR band B "
        "with a 3–16 kHz carrier, the 9 kHz receiver already smears the lines. "
        "Spreading does not reduce energy; max-hold then catches the extra "
        "peaks. The model reports about +1 dB, not a 5–10 dB win. Volunteer "
        "this; it is a sign you read the measurement, not a bug you hide.",
    ))
    story.append(P("<b>What do you need from us?</b>", "body_tight"))
    story.append(P(
        "A CSV of chamber results: the six design parameters, measured peak "
        "per band, and the limit you actually used. Tens of points, not "
        "thousands. That is enough to run the calibration path that is "
        "already written. Mentoring on whether the lumped cable story matches "
        "your fixtures is the second gift; it does not require us to pretend "
        "your data is in the training set.",
    ))

    story.append(P("Landmines", "h2"))
    story.append(simple_table(
        ["If you are tempted to say", "Say this instead"],
        [
            ["“AI-powered certification.”", "“A physics-informed pre-compliance risk indicator.”"],
            ["“Trained on EMC data.”", "“Trained on 5 000 simulated designs. No chamber data yet.”"],
            ["“99% accurate.”", "“About 1–2 dB vs our simulator from design parameters alone. Lab error is unknown.”"],
            ["“The ± is the uncertainty.”", "“The ± is disagreement among models of the same simulator. It is a floor.”"],
            ["“EN 12016 compliant.”", "“Below our assumed CISPR-shaped envelope, which is not the standard table.”"],
            ["“It models the whole EMC problem.”", "“Conducted 150 kHz–30 MHz on the motor-cable path. Not radiated, not immunity.”"],
            ["“Input filter will fix EMC.”", "“Input filter moves 50 Hz THD. It is excluded from the EMC model on purpose.”"],
        ],
        [70 * mm, 105 * mm],
    ))
    story.append(Spacer(1, 8))

    story.append(P("What a serious next step looks like", "h2"))
    story.append(bullets([
        "Replace the synthetic anchors with a limit line the company has the right to use, if they want named-standard scoring.",
        "Run calibrate.py on a few dozen chamber points from production-like cable lengths, including at least some 5–30 MHz fails.",
        "Keep monotone constraints. Do not fine-tune them away to chase a flattering residual.",
        "Report residual by band in public, including the ugly band.",
        "Leave certification where it belongs: in the lab report.",
    ]))

    # ============================================================== PART VI
    story.append(PageBreak())
    story.append(P("PART VI  ·  REFERENCE", "part"))
    story.append(P("Numbers you can quote if asked", "h1"))
    story.append(simple_table(
        ["Item", "Value"],
        [
            ["Training designs", "5 000 simulated (3 500 / 750 / 750 split)"],
            ["EMC sample rate", "200 MS/s, 2¹⁶ samples, 6 dwell segments, 9 kHz RBW"],
            ["PQ sample rate", "200 kS/s, 2¹⁴ samples, 50 Hz THD"],
            ["DC-link (model)", "565 V"],
            ["Cable capacitance", "100 pF/m to earth"],
            ["Shield", "up to 26 dB insertion loss + HF roll-off"],
            ["Model family", "XGBoost classifier + regressor per band, monotone constraints"],
            ["SHAP", "Exact tree pred_contribs on design-only margin models"],
            ["Ensemble", "5 bootstrap members for the score ±"],
            ["Score map", "50 + 50 tanh(margin_dB / 8), then 0.5·mean + 0.5·worst"],
            ["Tiers", "LOW ≥ 70, MODERATE 40–69, HIGH &lt; 40"],
            ["Full-model margin MAE", "0.13–0.55 dB vs simulator (sees the spectrum)"],
            ["Design-only margin MAE", "1.06 / 1.23 / 1.88 dB (A / B / C)"],
            ["Seed-to-seed simulator scatter", "~0.25 dB"],
            ["Stack", "Python FastAPI + React, reportlab PDFs"],
        ],
        [48 * mm, 127 * mm],
    ))
    story.append(Spacer(1, 10))

    story.append(P("Glossary", "h1"))
    glossary = [
        ("CISPR / EN 55011", "The family of emission measurement methods and industrial-equipment limits this curve is shaped like. CISPR 16-1-1 defines the 9 kHz receiver bandwidth used here."),
        ("Common mode (CM)", "The part of the voltage that is common to all motor phases versus earth. It drives conducted emissions on the cable screen / earth path."),
        ("Conducted emissions", "Noise measured on the power leads, typically 150 kHz–30 MHz, using a LISN and an EMI receiver."),
        ("dBµV", "Decibels relative to one microvolt. The unit on a conducted-emission plot. 0 dBµV = 1 µV across the LISN."),
        ("Design-only ablation", "Retraining the models with the spectrum hidden, so they must infer EMC from the six knobs. The honest skill figure."),
        ("dv/dt", "How fast the voltage changes at a switching edge, in volts per microsecond. Sets the high-frequency spectral corner."),
        ("EN 12016", "Product-family EMC standard for lifts, escalators and moving walks (immunity side, with emission practice around it). Named often; its tables are not in this tool."),
        ("LISN", "Line Impedance Stabilisation Network. Gives the measurement a defined 50 Ω view of the mains and dumps noise into the receiver."),
        ("Margin", "Limit minus emission, in dB. Positive = simulated headroom. Negative = simulated exceedance of the assumed curve."),
        ("Monotone constraint", "A hard rule on a boosted tree: this input may only push the prediction up, or only down. Here, the physics signs."),
        ("PWM", "Pulse-width modulation. How the inverter fakes a smooth motor voltage with a stream of DC-link pulses."),
        ("RBW", "Resolution bandwidth. How wide a frequency “bin” the receiver integrates. 9 kHz in this product."),
        ("SHAP", "A method for splitting one prediction into per-input contributions that add up. Used here in dB of margin."),
        ("Surrogate model", "A fast stand-in for a slower simulator, trained on the simulator’s own outputs."),
        ("THD", "Total harmonic distortion. How much extra energy sits in harmonics of 50 Hz. A power-quality number, not an EMC-limit number."),
        ("VFD / VVVF", "Variable-frequency (variable-voltage) drive. The inverter that runs the traction machine or escalator motor."),
        ("XGBoost", "A library for gradient-boosted trees. The specific ML method in this product, chosen because constraints and exact SHAP are native."),
    ]
    for term, meaning in glossary:
        story.append(P(f"<b>{term}.</b> {meaning}", "body_tight"))
    story.append(Spacer(1, 8))

    story.append(P("What this document is not", "h1"))
    story.append(P(
        "It is not a substitute for EN 12016, CISPR 11, CISPR 16, or any "
        "accredited test report. It is not a characterisation of any "
        "manufacturer’s shipped drive. Preset names are archetypes. Mentoring "
        "or industry review that informed the risk framing is not a product "
        "endorsement and is not validation against that organisation’s "
        "hardware. The software remains a simulation.",
    ))
    story.append(Spacer(1, 6))
    story.append(HairRule(CONTENT_W, RULE_STRONG, 0.8))
    story.append(Spacer(1, 8))
    story.append(P(
        "This is a simulation-based pre-compliance risk assessment intended to "
        "support early-stage design decisions. It does not replace accredited "
        "EMC certification testing (for example per EN 12016), and it does not "
        "predict certification outcomes.",
        "muted",
    ))
    story.append(P(
        "Regenerate this file from the repository with "
        "<font face='Courier'>python tools/company_briefing_pdf.py</font> "
        "inside the backend environment.",
        "small",
    ))

    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="EMC Advisor — company study briefing",
        author="EMC Advisor",
        subject="Briefing for presenting a virtual EMC pre-compliance advisor",
    )
    frame = Frame(
        PAGE_MARGIN,
        16 * mm,
        CONTENT_W,
        PAGE_H - 34 * mm,
        id="body",
        showBoundary=0,
    )
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=on_page))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf = build()
    OUT_PATH.write_bytes(pdf)
    print(f"Wrote {OUT_PATH} ({len(pdf) / 1024:.0f} kB)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
