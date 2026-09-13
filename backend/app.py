"""
app.py -- CertifAI HTTP API.

Endpoints
---------
    GET  /api/health           service and model-artifact status
    GET  /api/devices          preset device profiles + parameter metadata
    GET  /api/methodology      limit-curve assumptions, feature schema, constraints
    POST /api/predict          run an assessment for a device configuration
    POST /api/certificate      render an assessment result as a PDF

Design notes
------------
``/api/predict`` returns everything the results page needs in one response --
per-band verdicts, scores, the full spectrum trace and the risk attribution -- so
the frontend never has to make a second round trip to render a complete result.

``/api/certificate`` deliberately re-runs the prediction from the *parameters* in
the submitted payload rather than trusting the client-supplied scores. Otherwise a
caller could hand back an edited result object and receive a PDF asserting
whatever compliance score they chose, which for an assessment document is not an
acceptable trust boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from certificate import build_certificate, certificate_filename
from devices import DEVICE_PROFILE_BY_ID, list_profiles
from features import (
    BAND_FEATURE_SPECS,
    DESIGN_FEATURE_SPECS,
    EMC_BANDS,
    FEATURE_NAMES,
    HARMONIC_PROXIMITY_DB,
    LIMIT_CURVE_ANCHORS,
    LIMIT_CURVE_DESCRIPTION,
    MONOTONE_CONSTRAINTS_MARGIN,
    MONOTONE_CONSTRAINTS_RISK,
    RECEIVER_RBW_HZ,
    limit_dbuv,
)
from predictor import (
    DISCLAIMER_LONG,
    DISCLAIMER_SHORT,
    ModelNotTrainedError,
    load_models,
    predict,
)
from simulate import (
    N_SEGMENTS,
    PARAMETER_RANGES,
    PWM_CM_PENALTY_DB,
    PWM_MODULATION_LABELS,
    PWM_MODULATION_TYPES,
    SAMPLE_RATE_HZ,
    DeviceParameters,
)

logger = logging.getLogger("certifai")

app = FastAPI(
    title="CertifAI",
    version="1.0.0",
    description=(
        "Physics-informed EMC pre-compliance screening for elevator drive and "
        "controller electronics. Simulation-based; not an accredited certification "
        "service."
    ),
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# The frontend is served separately in development (Vite on 5173).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
PwmType = Literal["SPWM", "SVPWM", "DPWM", "RANDOM_SPWM"]


class DeviceParametersIn(BaseModel):
    """The six design inputs. Bounds mirror simulate.PARAMETER_RANGES exactly."""

    model_config = ConfigDict(extra="forbid")

    switching_frequency_khz: float = Field(..., ge=2.0, le=20.0)
    dv_dt_v_per_us: float = Field(..., ge=500.0, le=10000.0)
    cable_length_m: float = Field(..., ge=1.0, le=120.0)
    shielding_quality: float = Field(..., ge=0.0, le=1.0)
    load_current_a: float = Field(..., ge=5.0, le=200.0)
    pwm_modulation_type: PwmType = "SPWM"

    def to_domain(self) -> DeviceParameters:
        return DeviceParameters(**self.model_dump())


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: DeviceParametersIn
    device_id: Optional[str] = Field(
        default=None,
        description="Preset profile this configuration started from, if any.",
    )
    device_name: Optional[str] = Field(
        default=None, max_length=120,
        description="Display name for the assessed configuration.",
    )


class CertificateRequest(BaseModel):
    """Only the parameters are trusted; scores are recomputed server-side."""

    model_config = ConfigDict(extra="forbid")

    parameters: DeviceParametersIn
    device_id: Optional[str] = None
    device_name: Optional[str] = Field(default=None, max_length=120)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_device_name(
    device_id: Optional[str], device_name: Optional[str]
) -> Optional[str]:
    if device_name:
        return device_name
    if device_id and device_id in DEVICE_PROFILE_BY_ID:
        return DEVICE_PROFILE_BY_ID[device_id].name
    return None


def _run_prediction(
    parameters: DeviceParametersIn,
    device_id: Optional[str],
    device_name: Optional[str],
) -> Dict[str, Any]:
    try:
        return predict(
            parameters.to_domain(),
            device_id=device_id,
            device_name=_resolve_device_name(device_id, device_name),
        )
    except ModelNotTrainedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - unexpected inference failure
        logger.exception("prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Assessment failed: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Service status, including whether trained artifacts are available."""
    try:
        models = load_models()
        report = models.report or {}
        return {
            "status": "ok",
            "models_loaded": True,
            "artifact_version": models.artifact_version,
            "trained_at": report.get("generated_at"),
            "n_training_designs": report.get("n_samples"),
        }
    except ModelNotTrainedError as error:
        return {
            "status": "degraded",
            "models_loaded": False,
            "detail": str(error),
        }


@app.get("/api/devices")
def devices() -> Dict[str, Any]:
    """Preset device profiles plus the metadata needed to render the input form."""
    return {
        "devices": list_profiles(),
        "parameters": [
            {
                "key": rge.key,
                "label": rge.label,
                "unit": rge.unit,
                "min": rge.minimum,
                "max": rge.maximum,
                "step": rge.step,
                "default": rge.default,
                "description": rge.description,
            }
            for rge in PARAMETER_RANGES
        ],
        "pwm_modulation_types": [
            {
                "value": name,
                "label": PWM_MODULATION_LABELS[name],
                "cm_penalty_db": PWM_CM_PENALTY_DB[name],
            }
            for name in PWM_MODULATION_TYPES
        ],
        "disclaimer": DISCLAIMER_SHORT,
    }


@app.get("/api/methodology")
def methodology() -> Dict[str, Any]:
    """Everything needed to render the Methodology page, straight from the source.

    Served from the same constants the model was trained with, so the page cannot
    drift out of step with the code.
    """
    anchor_frequencies = [a[0] for a in LIMIT_CURVE_ANCHORS]
    try:
        report = (load_models().report or {})
    except ModelNotTrainedError:
        report = {}

    return {
        "bands": [
            {
                "key": band.key,
                "label": band.label,
                "f_low_hz": band.f_low_hz,
                "f_high_hz": band.f_high_hz,
                "limit_low_dbuv": float(limit_dbuv(band.f_low_hz)[0]),
                "limit_high_dbuv": float(limit_dbuv(band.f_high_hz)[0]),
            }
            for band in EMC_BANDS
        ],
        "limit_curve": {
            "anchors_hz_dbuv": [list(a) for a in LIMIT_CURVE_ANCHORS],
            "anchor_frequencies_hz": anchor_frequencies,
            "description": LIMIT_CURVE_DESCRIPTION,
            "is_synthetic": True,
            "provenance": (
                "Anchored on publicly described CISPR 11 Group 1 Class A quasi-peak "
                "conducted-emission levels (79 dBuV in the 150-500 kHz region, "
                "73 dBuV above it), which EN 61000-6-4 and lift-sector practice "
                "broadly follow. The smooth tapers between anchors are additions by "
                "this tool. EN 12016's normative tables are copyrighted and not "
                "publicly redistributable, so this curve is a stand-in and a pass "
                "against it is not a statement about EN 12016 conformity."
            ),
        },
        "simulation": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "n_dwell_segments": N_SEGMENTS,
            "receiver_rbw_hz": RECEIVER_RBW_HZ,
            "harmonic_proximity_db": HARMONIC_PROXIMITY_DB,
        },
        "features": {
            "count": len(FEATURE_NAMES),
            "design": [
                {
                    "name": spec.name,
                    "risk_sign": spec.risk_sign,
                    "rationale": spec.rationale,
                }
                for spec in DESIGN_FEATURE_SPECS
            ],
            "spectral": [
                {
                    "name": spec.name,
                    "risk_sign": spec.risk_sign,
                    "rationale": spec.rationale,
                }
                for spec in BAND_FEATURE_SPECS
            ],
        },
        "monotone_constraints": {
            "risk": list(MONOTONE_CONSTRAINTS_RISK),
            "margin": list(MONOTONE_CONSTRAINTS_MARGIN),
            "explanation": (
                "Each model input has a known physical direction of influence on "
                "failure risk. Those directions are passed to XGBoost as monotone "
                "constraints, so the fitted trees are structurally unable to learn a "
                "physically impossible relationship -- for example that adding cable "
                "screening increases emissions. Signs are stated with respect to "
                "failure risk; the margin regressors, which predict headroom, use the "
                "negated vector."
            ),
        },
        "validation": {
            "metric_semantics": report.get("metric_semantics"),
            "n_samples": report.get("n_samples"),
            "n_test": report.get("n_test"),
            "band_metrics": report.get("band_metrics"),
            "design_only_ablation": report.get("design_only_ablation"),
            "seed_stability": report.get("seed_stability"),
            "monotonicity_audit": report.get("monotonicity_audit"),
        },
        "disclaimer": DISCLAIMER_SHORT,
        "disclaimer_long": DISCLAIMER_LONG,
    }


@app.post("/api/predict")
def run_prediction(request: PredictRequest) -> Dict[str, Any]:
    """Simulate, extract features, run inference and score one configuration."""
    return _run_prediction(request.parameters, request.device_id, request.device_name)


@app.post("/api/certificate")
def certificate(request: CertificateRequest) -> Response:
    """Render the Virtual Pre-Compliance Assessment PDF.

    The assessment is recomputed here from the submitted parameters, so the PDF
    always reflects what the model actually predicts for that configuration rather
    than whatever numbers the client sent.
    """
    result = _run_prediction(
        request.parameters, request.device_id, request.device_name
    )
    try:
        pdf = build_certificate(result)
    except Exception as error:  # pragma: no cover - rendering failure
        logger.exception("certificate rendering failed")
        raise HTTPException(
            status_code=500, detail=f"Could not render the PDF: {error}"
        ) from error

    filename = certificate_filename(result)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/presets/{device_id}")
def preset(device_id: str) -> Dict[str, Any]:
    """A single preset profile, for deep links into the parameter page."""
    profile = DEVICE_PROFILE_BY_ID.get(device_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown device '{device_id}'")
    return profile.as_dict()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
