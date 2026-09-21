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
  input_filter_quality: number
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
  framing?: string
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

export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH'

export interface SignalTrace {
  key: string
  label: string
  unit: string
  timescale: string
  description: string
  time_ms: number[]
  values: number[]
}

export interface HarmonicPeak {
  order: number
  frequency_hz: number
  amplitude: number
  percent_of_fundamental: number
}

export interface PowerQualityResult {
  key: string
  label: string
  thd_percent: number
  fundamental_hz: number
  dominant_orders: number[]
  dominant_statement: string
  peaks: HarmonicPeak[]
}

export interface ShapRow {
  parameter: string
  label: string
  shap: number
  raises_risk: boolean
}

export interface SensitivityBar {
  parameter: string
  label: string
  unit?: string
  delta_score: number
  direction?: string
}

export interface Countermeasure {
  parameter: string
  label: string
  suggestion: string
  current_value: number | string
  suggested_value: number | string
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
  framing: string
  risk_level: RiskLevel
  risk_label: string
  risk_copy: string
  verdict: RiskLevel | 'PASS' | 'FAIL'
  compliance_score: number
  risk_score: number
  risk_score_low: number
  risk_score_high: number
  risk_score_plus_minus: number
  confidence_score: number
  confidence_label: string
  confidence_ceiling: number
  confidence_note: string
  bands: BandResult[]
  worst_band: string
  top_risk_factor: RiskFactor
  shap: { band: string; note: string; contributions: ShapRow[]; unit?: string }
  countermeasures: Countermeasure[]
  sensitivity: { note: string; bars: SensitivityBar[] }
  signals: SignalTrace[]
  power_quality: PowerQualityResult[]
  spectrum: SpectrumTrace
  simulation_diagnostics: Record<string, number>
  model_info: {
    artifact_version: string
    feature_count: number
    monotone_constraints: number[]
    simulation_consistency: SimulationConsistency
    ensemble_size?: number
  }
  limit_curve: {
    anchors_hz_dbuv: number[][]
    description: string
    is_synthetic: boolean
  }
  disclaimer: string
  disclaimer_long: string
}

export interface BandDiff {
  key: string
  label: string
  baseline_margin_db: number
  candidate_margin_db: number
  delta_margin_db: number
}

export interface CompareResult {
  baseline: PredictionResult
  candidate: PredictionResult
  delta_risk_score: number
  band_diff: BandDiff[]
}

export interface AssessmentHistoryEntry {
  score: number
  plusMinus: number
  level: RiskLevel
  name: string
  at: string
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
  framing?: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  models_loaded: boolean
  artifact_version?: string
  trained_at?: string
  n_training_designs?: number
  detail?: string
}

export interface ValidationFeatureRow {
  label: string
  checked: number
  violations: number
  risk_sign: number
}

export interface ValidationResponse {
  generated_at: string
  artifact_version: string
  passed: boolean
  headlines: {
    monotonicity: string
    shap_additivity: string
  }
  monotonicity: {
    passed: boolean
    headline: string
    n_configs: number
    n_steps: number
    n_checks: number
    n_violations: number
    consistent_percent: number
    per_feature: Record<string, ValidationFeatureRow>
    note: string
  }
  shap_additivity: {
    passed: boolean
    headline: string
    n_designs: number
    n_checks: number
    mean_abs_error_db: number
    max_abs_error_db: number
    mean_abs_error_display: string
    max_abs_error_display: string
    per_band: {
      band: string
      n: number
      mean_abs_error_db: number
      max_abs_error_db: number
    }[]
    note: string
  }
  simulation_consistency: {
    metric_semantics: string | null
    n_test: number | null
    design_only_ablation: {
      note?: string
      band_metrics?: {
        band_label: string
        classifier_balanced_accuracy: number
        margin_mae_db: number
      }[]
    }
  }
  note: string
  suite_path?: string
}

export interface TradeoffPoint {
  switching_frequency_khz: number
  emc_risk_score: number
  ripple_cost: number
  acoustic_risk: number
  ripple_peak_to_peak_a: number
  predicted_margins_db: Record<string, number>
  pareto_ripple: boolean
  pareto_acoustic: boolean
  is_current: boolean
}

export interface TradeoffResponse {
  caption: string
  formulas: {
    ripple_cost: string
    acoustic_risk: string
    ripple_ref_a: number
    l_motor_h: number
    v_dc_v: number
    audible_center_khz: number
    audible_scale_khz: number
  }
  held: Omit<DeviceParameters, 'switching_frequency_khz'>
  sweep: {
    f_min_khz: number
    f_max_khz: number
    n_points: number
    seed: number
  }
  points: TradeoffPoint[]
  diagnostics: {
    n_pareto_ripple: number
    n_pareto_acoustic: number
    ripple_trades_off_with_emc: boolean
    acoustic_trades_off_with_emc: boolean
    agrees_with_carrier_countermeasure: boolean
    pareto_note: string
  }
}
