"""
calibrate.py -- EXTENSION POINT (not implemented).

Everything CertifAI currently knows comes from the synthetic physics model in
``simulate.py``. This module is the designated place to correct the models towards
real measured data once any exists, and it is deliberately left as a stub: a
half-built calibration path is worse than an obviously absent one, because it
invites the tool to be trusted more than the evidence supports.

Why a separate module rather than retraining
--------------------------------------------
Real EMC pre-compliance data arrives in tiny quantities -- tens of measurements,
not thousands -- because each point costs chamber or LISN bench time. That rules
out simply appending it to the 5000 synthetic samples in ``train_model.py``, where
it would be numerically drowned. The established approach for this situation is
residual correction: keep the physics-derived model as the prior, and learn a
small, heavily regularised correction from simulated to measured levels on top of
it.

What the pipeline is designed to accept
---------------------------------------
The training pipeline was structured so this can be added without touching the
simulator, the feature extractor or the API:

  * ``features.FEATURE_NAMES`` and ``features.MONOTONE_CONSTRAINTS_RISK`` are the
    single source of truth for the feature schema, so a correction model can be
    built over the identical feature space with the identical physical constraints.
  * ``train_model.py`` saves the fitted ``StandardScaler`` alongside the models,
    so measured samples can be mapped into the same feature space that the
    synthetic models were fitted in.
  * ``predictor.load_models`` reads a single artifact bundle and validates its
    feature schema, so a calibrated bundle can be swapped in without changes
    elsewhere.
  * ``predictor.ModelBundle.margin_rmse_db`` feeds the confidence calculation, so
    a calibrated model's measured error automatically propagates into the
    confidence figures the UI and the PDF report.

Expected input format
---------------------
A CSV with one row per measured configuration:

    switching_frequency_khz, dv_dt_v_per_us, cable_length_m, shielding_quality,
    load_current_a, pwm_modulation_type,
    measured_peak_dbuv_band_a, measured_peak_dbuv_band_b, measured_peak_dbuv_band_c,
    limit_dbuv_band_a, limit_dbuv_band_b, limit_dbuv_band_c

with the measurement made using a CISPR 16-1-1 compliant receiver, 9 kHz
resolution bandwidth, quasi-peak detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def calibrate_from_measurements(
    measurements_csv: Path,
    *,
    model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    max_correction_db: float = 12.0,
) -> Dict[str, Any]:
    """Fine-tune the synthetic models against a small real measured dataset.

    NOT IMPLEMENTED. This is the documented extension point described in the module
    docstring; calling it raises ``NotImplementedError`` by design.

    The intended implementation, for whoever picks this up:

    1.  Load the measured CSV and rebuild each row's ``DeviceParameters``.
    2.  Re-run ``simulate_device`` + ``extract_features`` for every measured
        configuration to obtain the simulator's own prediction for it. This is the
        paired synthetic-vs-measured dataset the correction is learned from.
    3.  Fit a per-band **residual model** on ``measured_margin - predicted_margin``.
        With only tens of samples this must be strongly regularised -- ridge
        regression on the six design parameters, or an isotonic fit on the single
        strongest predictor, rather than another gradient-boosted ensemble.
        Preserve the physics constraints: the correction must not be allowed to
        invert a known monotonic relationship.
    4.  Clamp the correction to +/- ``max_correction_db``. A residual larger than
        that means the simulator is wrong about the *mechanism*, not merely
        miscalibrated, and the right response is to fix ``simulate.py`` rather
        than to paper over it in post-processing.
    5.  Re-estimate the per-band margin RMSE by leave-one-out cross-validation on
        the measured set, and write it into ``margin_rmse_db``. This is the step
        that makes the confidence scores honest: once real data exists, the
        confidence figure should reflect error against *hardware*, not against the
        simulator.
    6.  Save a new bundle with ``calibration`` metadata recording the sample count,
        date, measuring equipment and residual statistics, so that a calibrated
        result can always be distinguished from an uncalibrated one, in the API
        response and on the PDF.

    Until this exists, every figure the tool reports is relative to its own
    simulation, and the UI says so.
    """
    raise NotImplementedError(
        "calibrate_from_measurements is an intentional extension point and is not "
        "implemented. See the module docstring in calibrate.py for the design, the "
        "expected CSV format, and the reasoning behind the residual-correction "
        "approach."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "calibrate.py is an unimplemented extension point. Read the module "
        "docstring for what it is for and how it is meant to work."
    )
