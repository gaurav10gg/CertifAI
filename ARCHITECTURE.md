# Architecture

EMC Advisor (Virtual EMC Pre-Compliance Advisor) estimates **conducted EMC risk** for elevator VFD electronics from a handful of design parameters. It is a pre-compliance **risk indicator** for design decisions. It does not predict EN 12016 certification outcomes.

## Pipeline

```
parameters
  → simulate.py     five time-domain signals (HF EMC capture + 50 Hz PQ record)
  → features.py     3-band conducted-emission margins + motor/input THD
  → XGBoost         monotone-constrained P(fail) + margin per band
  → predictor.py    risk score, ensemble ±, SHAP, sensitivity, countermeasures
  → FastAPI / UI / PDF
```

**Simulation (two timescales).** Conducted-emission analysis uses a 200 MS/s, six-segment max-hold capture of common-mode current into a 50 Ω LISN — the quantity an EMI receiver would see. Power-quality analysis uses a cheaper 200 kS/s × 2¹⁴ record so 50 Hz THD is actually resolvable. The 328 µs EMC window cannot resolve mains harmonics; mixing the two timescales is deliberate.

Five explorer traces: DC-link voltage, motor line-to-line PWM, RL-filtered motor current, common-mode voltage, 6-pulse input current. `input_filter_quality` is a **PQ-only** lever (line reactor). It is not an XGBoost feature, so it cannot silently pretend to fix motor-cable EMC.

**Features.** Eighteen numbers enter the models: six design parameters (switching frequency, dv/dt, cable length, shielding, load current, PWM CM penalty) plus four spectral descriptors per band. Motor/input THD and the top-five harmonic peaks are reported beside the models, not stuffed into them — they answer a power-quality question, not an EMC-limit question.

**Models.** Per band: XGBClassifier + XGBRegressor with `monotone_constraints` matching the sign of each feature on failure risk. An optional bootstrap ensemble of five lighter margin regressors supplies the ± band on the risk score. SHAP is exact tree `pred_contribs`, not a KernelSHAP approximation. Local sensitivity is a finite-difference of the risk score from the current design.

**Scoring.** Margin → 0–100 via tanh (0 dB → 50). Overall score blends mean and worst band. Tiers: LOW 70–100, MODERATE 40–69, HIGH 0–39. Higher score = more simulated headroom = lower risk.

**Validation (Phase 7).** `python tools/run_validation.py` writes `models/validation_suite.json`. The Model Validation page (`GET /api/validation`) shows the measured headlines: a 200-configuration monotonicity sweep of the six constrained design parameters, and SHAP additivity (`sum(SHAP) + bias ≈ predicted margin`) on the design-only regressors. Pytest re-runs a smaller probe of both checks. None of this is lab accuracy.

## Assumptions that dominate error

1. The limit curve is a **synthetic** EN-12016-style envelope, not the copyrighted table.
2. Every accuracy figure is **simulation-consistency**, not lab accuracy. No chamber data is in the training loop.
3. Cable, shield and rectifier models are lumped / first-order. They capture direction of change, not fixture parasitics.
4. The ensemble ± is disagreement among models trained on the same simulator. It is a floor on true uncertainty.

## What real KONE data would change

`calibrate.py` already fits a clamped residual (measured − simulated margin) on the six design features and writes a new joblib version. A few dozen chamber points would:

- shift the absolute score toward lab reality without dropping monotonicity;
- let the ± band be an empirical residual instead of ensemble spread;
- show which bands the lumped cable model misplaces (likely 5–30 MHz).

It would **not** replace the physics prior, and it would not turn this tool into a certification oracle. Accredited measurement remains the only certification path.
