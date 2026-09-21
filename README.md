# EMC Advisor

**Virtual EMC Pre-Compliance Advisor for elevator VFD electronics.**

A physics-informed **risk indicator** for conducted EMC on elevator drive
electronics. It estimates a three-tier risk level (LOW / MODERATE / HIGH) from
design parameters, five simulated waveforms, and monotonically constrained
XGBoost models. It exists to answer a design question that normally waits for a
chamber booking: *is this cable run, switching frequency and shield specification
going to be a problem?*

> **This is a simulation.** It does not predict EN 12016 certification outcomes,
> which require accredited lab measurement. No measured hardware data is used in
> the training loop, and the limit curve is a documented assumption. See
> [Honesty](#honesty-about-what-this-is) and [`ARCHITECTURE.md`](ARCHITECTURE.md).
> A plain-language tour of the whole product, with every physics term defined,
> is [`docs/EMC_Advisor_Explained.pdf`](docs/EMC_Advisor_Explained.pdf)
> (regenerate with `python tools/explained_pdf.py` from `backend/`).

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
python train_model.py --ensemble-only   # optional: 5-model uncertainty ensemble
uvicorn app:app --port 8000
```

`train_model.py` writes `backend/models/certifai_models.joblib`. The API returns
`503` on every prediction endpoint until it exists.

### Frontend

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

### Checks

```bash
cd backend
python -m pytest
python tools/run_validation.py          # 200-config monotonicity + SHAP identity
python tools/api_check.py               # needs a running API
```

---

## What it does

1. **Physics simulation** (`simulate.py`) — trapezoidal PWM, common-mode
   `v_cm = (v_a+v_b+v_c)/3`, cable resonances, shield, plus four companion
   waveforms: DC-link voltage, motor line-to-line PWM, RL-filtered motor current,
   6-pulse input current. Switching frequency is 3–16 kHz.
2. **Frequency analysis** (`features.py`) — CISPR 9 kHz RBW conducted-emission
   margins in three bands (EMC risk), and explicit THD % plus top-5 harmonics on
   motor/input current (power quality). Those two are kept separate on purpose.
3. **Physics-informed ML** (`train_model.py` / `predictor.py`) — XGBoost with
   `monotone_constraints`, exact tree SHAP, a 5-member ensemble for the ± band
   on the risk score, and a local sensitivity tornado.
4. **Dashboard + PDF** — risk badge, uncertainty, SHAP waterfall, tornado,
   tabbed signal explorer, Compare Mode, last-three-run sparkline, and a
   “Virtual EMC Pre-Compliance Report”.

Risk score 70–100 = LOW, 40–69 = MODERATE, 0–39 = HIGH. Higher score means more
simulated headroom, not a lab pass.

---

## Honesty about what this is

**The limit curve is synthetic.** EN 12016 tables are not redistributable. The
curve is a CISPR-style envelope, flagged as an assumption in `features.py`, the
methodology page, and the PDF. Direction and size of a design change are the
reliable output; an absolute tier is only as good as that curve.

**Accuracy figures are simulation-consistency scores.** They measure agreement
with this tool's own physics on held-out synthetic designs — never with hardware.

**The ensemble ± is not lab uncertainty.** It is spread across models trained on
the same simulator, and is therefore a floor on the true error.

**`calibrate.py` is implemented but untested on chamber data.** It will fit a
clamped residual from a CSV of real measurements. Until that CSV exists, every
figure remains relative to the simulator.

---

## Documentation

| File | Who it is for |
|---|---|
| [`docs/EMC_Advisor_Explained.pdf`](docs/EMC_Advisor_Explained.pdf) | Anyone. Every physics term defined in ordinary language. Regenerate with `python tools/explained_pdf.py` from `backend/`. |
| [`docs/EMC_Advisor_Company_Briefing.pdf`](docs/EMC_Advisor_Company_Briefing.pdf) | Internal study briefing for presenting the product. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Engineers: modules, data flow, honesty constraints. |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service and model-artifact status |
| `GET` | `/api/devices` | Preset profiles, parameter bounds, PWM options |
| `GET` | `/api/methodology` | Assumptions, feature schema, validation report |
| `GET` | `/api/validation` | Phase 7 suite: monotonicity (200 configs) + SHAP additivity |
| `POST` | `/api/predict` | Full risk assessment |
| `POST` | `/api/compare` | Side-by-side of two configurations |
| `POST` | `/api/sensitivity` | Physics-level tornado (re-simulates) |
| `POST` | `/api/certificate` | Virtual EMC Pre-Compliance Report PDF |
| `POST` | `/api/tradeoff` | Sweep switching frequency for EMC vs acoustic vs ripple |
| `GET` | `/api/presets/{id}` | A single preset's parameters |

`POST /api/certificate` accepts only *parameters* and recomputes the assessment
server-side.

## Design

Strictly monochrome, with one green and one red reserved for risk-tier / band
exceedance cues — never as the only signal, always beside a label. Dark mode
toggles a `.dark` class on `<html>` before first paint.
