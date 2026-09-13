/**
 * types.ts -- shapes returned by the CertifAI backend.
 *
 * These mirror the response bodies built in backend/predictor.py and
 * backend/app.py. Kept hand-written rather than generated so the field-level
 * comments explaining what each number actually means live next to the type.
 */

export type PwmModulationType = 'SPWM' | 'SVPWM' | 'DPWM' | 'RANDOM_SPWM'

export interface DeviceParameters {
  switching_frequency_khz: number
  dv_dt_v_per_us: number
  cable_length_m: number
  shielding_quality: number
  load_current_a: number
  pwm_modulation_type: PwmModulationType
}

export interface ParameterSpec {
  key: keyof Omit<DeviceParameters, 'pwm_modulation_type'>
  label: string
  unit: string
  min: number
  max: number
  step: number
  default: number
  description: string
}

export interface PwmOption {
  value: PwmModulationType
  label: string
  /** Modelled common-mode penalty in dB, relative to plain sinusoidal PWM. */
  cm_penalty_db: number
}

export interface DeviceProfile {
  id: string
  name: string
  category: string
  summary: string
  highlights: string[]
  icon: string
  parameters: DeviceParameters
}

export interface DevicesResponse {
  devices: DeviceProfile[]
  parameters: ParameterSpec[]
  pwm_modulation_types: PwmOption[]
  disclaimer: string
}

export interface BandResult {
  key: string
  label: string
  f_low_hz: number
  f_high_hz: number
  /** Verdict from the margin regressor: predicted headroom above zero. */
  passes: boolean
  fail_probability: number
  predicted_margin_db: number
  band_score: number
  /** Conservative per-band margin error used for the confidence figure. */
  margin_uncertainty_db: number
  /** The simulator's own measurement, for comparison against the prediction. */
  simulated_margin_db: number
  simulated_peak_dbuv: number
  simulated_peak_frequency_hz: number
  limit_at_peak_dbuv: number
  worst_frequency_hz: number
  harmonic_count: number
  thd_score_db: number
}

export interface RiskFactor {
  parameter: string
  label: string
  unit: string
  current_value: number | string
  suggested_value: number | string
  attribution_score: number
  driving_band: string
  driving_band_label: string
  statement: string
  suggestion: string
  ranking: { parameter: string; attribution_score: number }[]
}

export interface SpectrumTrace {
  frequency_hz: number[]
  emission_dbuv: number[]
  limit_dbuv: number[]
}

export interface BandConsistency {
  band_label: string
  balanced_accuracy: number | null
  roc_auc: number | null
  margin_mae_db: number
  margin_r2: number
}

export interface SimulationConsistency {
  note: string
  n_test_designs: number | null
  n_training_designs: number | null
  mean_balanced_accuracy: number | null
  mean_margin_mae_db: number | null
  per_band: BandConsistency[]
  design_only_ablation: {
    note: string
    mean_balanced_accuracy: number | null
    mean_margin_mae_db: number | null
    per_band: BandConsistency[]
  }
  seed_stability: {
    simulator_margin_sd_db: number
    model_margin_sd_db: number
    variance_reduction_factor: number | null
  } | null
}

export interface ParameterDisplayRow {
  key: string
  label: string
  unit: string
  value: number | string
  formatted: string
}

export interface PredictionResult {
  generated_at: string
  device_id: string | null
  device_name: string
  parameters: DeviceParameters
  parameter_display: ParameterDisplayRow[]
  verdict: 'PASS' | 'FAIL'
  compliance_score: number
  confidence_score: number
  confidence_label: string
  confidence_ceiling: number
  confidence_note: string
  bands: BandResult[]
  worst_band: string
  top_risk_factor: RiskFactor
  spectrum: SpectrumTrace
  simulation_diagnostics: Record<string, number>
  model_info: {
    artifact_version: string
    feature_count: number
    monotone_constraints: number[]
    simulation_consistency: SimulationConsistency
  }
  limit_curve: {
    anchors_hz_dbuv: number[][]
    description: string
    is_synthetic: boolean
  }
  disclaimer: string
  disclaimer_long: string
}

export interface FeatureSpecInfo {
  name: string
  risk_sign: number
  rationale: string
}

export interface MethodologyResponse {
  bands: {
    key: string
    label: string
    f_low_hz: number
    f_high_hz: number
    limit_low_dbuv: number
    limit_high_dbuv: number
  }[]
  limit_curve: {
    anchors_hz_dbuv: number[][]
    anchor_frequencies_hz: number[]
    description: string
    is_synthetic: boolean
    provenance: string
  }
  simulation: {
    sample_rate_hz: number
    n_dwell_segments: number
    receiver_rbw_hz: number
    harmonic_proximity_db: number
  }
  features: {
    count: number
    design: FeatureSpecInfo[]
    spectral: FeatureSpecInfo[]
  }
  monotone_constraints: {
    risk: number[]
    margin: number[]
    explanation: string
  }
  validation: {
    metric_semantics: string | null
    n_samples: number | null
    n_test: number | null
    band_metrics:
      | {
          band_label: string
          fail_rate: number
          classifier_balanced_accuracy: number
          classifier_roc_auc: number | null
          margin_mae_db: number
          margin_rmse_db: number
          margin_r2: number
        }[]
      | null
    design_only_ablation: {
      note: string
      feature_names: string[]
      band_metrics: {
        band_label: string
        classifier_balanced_accuracy: number
        classifier_roc_auc: number | null
        margin_mae_db: number
        margin_rmse_db: number
        margin_r2: number
      }[]
    } | null
    seed_stability: {
      simulator_margin_sd_db: number
      model_margin_sd_db: number
      variance_reduction_factor: number | null
    } | null
    monotonicity_audit:
      | { feature: string; risk_sign: number; respects_constraint: boolean }[]
      | null
  }
  disclaimer: string
  disclaimer_long: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  models_loaded: boolean
  artifact_version?: string
  trained_at?: string
  n_training_designs?: number
  detail?: string
}
