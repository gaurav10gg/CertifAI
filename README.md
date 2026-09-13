# CertifAI

**Virtual EMC pre-compliance assessment for elevator drive electronics.**

CertifAI predicts whether an elevator drive's conducted emissions would pass an
EN 12016-style test, from six design parameters, and produces a downloadable
assessment document. It exists to answer a question that normally has to wait for
a chamber booking: *is this cable run, switching frequency and shield
specification going to be a problem?*

The prediction chain is a physics simulation, not a lookup table: the tool
synthesises the inverter's switching waveform, derives the common-mode emission
spectrum through a receiver-emulated FFT, and evaluates gradient-boosted models
that are **structurally forbidden** from learning any relationship that
contradicts electromagnetics.

> **This is a simulation.** No measured hardware data is used anywhere in the
> pipeline, and the limit curve is a documented assumption rather than published
> data. See [Honesty about what this is](#honesty-about-what-this-is).

---

## Quick start

Two processes: a Python API on port 8000, and a Vite dev server on port 5173 that
proxies `/api` to it.

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

python train_model.py           # ~3 min: generates 5000 designs and trains
uvicorn app:app --port 8000
```

`train_model.py` writes `backend/models/certifai_models.joblib`. The API returns
`503` on every prediction endpoint until it exists, and `GET /api/health` reports
`models_loaded: false`, so a missing artifact fails loudly rather than silently
returning garbage.

### Frontend

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

### Verifying it works

```bash
cd backend
python tools/api_check.py       # sweeps each parameter through the live API
python train_model.py --report-only
```

`api_check.py` is the fastest end-to-end confidence check: it sweeps one
parameter at a time and prints the resulting scores, which should move
monotonically in the physically correct direction every time.

---

## How it works

### 1. Physics simulation — `backend/simulate.py`

Three inverter legs are switched with the chosen PWM strategy at the chosen
carrier frequency. From the leg voltages the model forms the **common-mode
voltage** `(v_a + v_b + v_c)/3` — a `V_DC/3` staircase stepping six times per
carrier period, and the quantity that actually drives conducted emissions.

Edges are trapezoidal with a rise time set by `dv/dt`, which fixes the corner of
the spectral envelope at `1/(π·t_rise)`. The common-mode displacement current
`i = C·dv/dt` (100 pF per metre of cable) is then shaped by the cable's
transmission-line resonances — a quarter-wave mode plus its 3rd and 5th
harmonics, modelling all three rather than only the fundamental, because a single
resonance slides out of the top band as the cable lengthens and would make the
response non-monotone. Finally the shield attenuates, load current scales, and a
randomised noise floor is summed in.

**Receiver dwell.** A single short capture sees one slice of the 50 Hz output
cycle, and the resulting duty-cycle variance swamps the parameter effects
entirely (this was measured: ±7–14 dB of scatter, non-monotone everywhere). The
simulator instead takes six short records spread across one fundamental period
and max-holds them — which is what a real EMI receiver does when it dwells on a
frequency.

### 2. Spectral analysis — `backend/features.py`

Segments are windowed, transformed, max-held, and power-integrated across a 9 kHz
resolution bandwidth (CISPR 16-1-1 band B) to produce the trace a test house
would read. Per band: peak amplitude, RMS amplitude, count of spectral lines
within 10 dB of the limit, and a THD-like noisiness score. With the six design
parameters that gives **18 features**.

### 3. Physics-informed models — `backend/train_model.py`

Per band, two XGBoost models — a pass/fail classifier and a continuous margin
regressor — both trained with `monotone_constraints`. The constraint vector is
the physics-informed part of this project and is written out explicitly, feature
by feature, with the reason for each sign:

| Parameter | Constraint | Reason |
|---|---|---|
| Switching frequency | `+1` | Spectral line amplitude scales with pulse repetition frequency |
| `dv/dt` | `+1` | Faster edges push the envelope's second corner upward |
| Cable length | `+1` | Parasitic capacitance to earth scales with length |
| Shielding quality | `−1` | A better screen can only attenuate |
| Load current | `+1` | Emission amplitude scales with switched current |
| PWM CM penalty | `+1` | A larger modelled penalty can only raise emissions |

The model is *forbidden* from learning that a longer cable reduces emissions,
even if a pocket of training data suggests it. This is enforced structurally at
training time, not checked afterwards — though it is also audited afterwards, by
sweeping each parameter across its range.

### 4. Inference — `backend/predictor.py`

Compliance score maps each band's margin through a saturating function (0 dB →
50), then blends the three-band mean with the worst band in equal parts, so a
design cannot score well by passing two bands and failing the third. Confidence
is the probability each verdict survives the model's own measured margin error,
reduced on classifier/regressor disagreement and capped below 100. Risk
attribution uses exact tree SHAP (`pred_contribs=True`), weighted by how far each
parameter already sits toward its risky extreme.

---

## Honesty about what this is

Four claims this project deliberately does **not** make.

**The limit curve is synthetic.** EN 12016 limit data is not publicly available.
The curve is a plausible CISPR-style envelope anchored at 79 dBµV @ 150 kHz →
73 dBµV @ 500 kHz–5 MHz → 70 dBµV @ 30 MHz, and it is flagged as an assumption in
`features.py`, in the methodology page, and in the PDF. What the tool tells you
reliably is the *direction and size* of a change; an absolute pass/fail is only
as good as that assumed curve.

**The accuracy figures are simulation-consistency scores.** They measure
agreement between the models and this tool's own physics simulation on held-out
synthetic designs — never agreement with measured hardware. They are labelled
that way in the code, the API, the UI and the PDF.

**The headline metrics are flattered by construction.** Balanced accuracy above
0.98 looks like leakage. It is not, but it is close to tautological: band peak
level is nearly a sufficient statistic for the margin label. So
`design_only_ablation()` retrains on the six design parameters alone — the harder
problem of predicting compliance from the design — giving 0.94–0.97 balanced
accuracy and 1.1–1.9 dB MAE. **Both** numbers are surfaced everywhere.

**The surrogate does not denoise the simulator.** An earlier version of this
README claimed it did. `audit_seed_stability()` measured 0.25 dB (simulator) vs
0.24 dB (surrogate) — no reduction. The measured negative result is reported in
the methodology page rather than quietly dropped, and the surrogate's real value
is stated as what it is: speed, and a calibration surface for real data.

The output document is titled **"Virtual Pre-Compliance Assessment"**, never
"Certificate of Compliance", and carries the disclaimer that it does not replace
accredited EMC certification testing.

### Extending to real measurements

`backend/calibrate.py` is a documented stub, deliberately unimplemented rather
than faked. It describes fitting a residual correction — measured minus predicted
margin, as a function of the design parameters — so real chamber data can improve
the tool without discarding the physics or the monotonicity guarantees.

---

## Project structure

```
backend/
  simulate.py        Trapezoidal PWM waveform synthesis, cable/shield/noise model
  features.py        FFT, receiver emulation, per-band features, limit curve
  train_model.py     Dataset generation, training, validation, ablation, audits
  predictor.py       Inference: scores, confidence, SHAP risk attribution
  devices.py         Six tuned preset device profiles
  certificate.py     PDF assessment document (reportlab + matplotlib)
  calibrate.py       Extension point for real-measurement calibration (stub)
  app.py             FastAPI application
  models/            Trained artifacts (generated)
  tools/
    api_check.py     Parameter sweeps against a running API
    sweep_check.py   Simulator-level monotonicity and population checks
    tune_presets.py  Preset parameter evaluation

frontend/
  src/pages/         DeviceSelect, ParameterInput, Results, Methodology
  src/components/    Card, Slider, ScoreDisplay, SpectrumChart, BandBreakdown,
                     RiskCallout, CertificateButton, Header, Stepper, Modal, ...
  src/api/           Typed client mirroring the backend responses
  src/theme/         Design tokens and dark-mode hook
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service and model-artifact status |
| `GET` | `/api/devices` | Preset profiles, parameter bounds, PWM options |
| `GET` | `/api/methodology` | Limit-curve assumptions, feature schema, validation report |
| `POST` | `/api/predict` | Full assessment: scores, per-band verdicts, spectrum, risk |
| `POST` | `/api/certificate` | Assessment PDF |
| `GET` | `/api/presets/{id}` | A single preset's parameters |

`POST /api/certificate` accepts only *parameters* and recomputes the assessment
server-side. A caller cannot hand back an edited result object and receive a PDF
asserting a score the model never produced.

## Design

Strictly monochrome, with one green and one red reserved for pass/fail — and
never as the only signal, always alongside a glyph and a word. Semantic tokens
(`paper`, `ink`, `line`, `accent`, `pass`, `fail`) are defined as RGB channel
triplets in `src/theme/tokens.css` and switched by a `.dark` class on `<html>`,
applied before first paint so there is no light flash on load.
