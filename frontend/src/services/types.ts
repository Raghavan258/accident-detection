export type MetricSlice = {
  precision_crash?: number
  recall_crash?: number
  f1_crash?: number
  average_precision_crash?: number
  false_positives?: number
  false_negatives?: number
}

export type Confusion = {
  labels: string[]
  matrix: number[][]
}

export type ExperimentRun = {
  run_id: string
  run_name: string | null
  status: string
  start_time: number | null
  end_time: number | null
  params: Record<string, string>
  validation: MetricSlice
  test: MetricSlice
  validation_confusion_matrix: Confusion | null
  test_confusion_matrix: Confusion | null
}

export type RegistryVersion = {
  version: string
  run_id: string
  tags: Record<string, string>
  creation_timestamp?: number
  last_updated_timestamp?: number
}

export type StatusPayload = {
  generated_at?: string
  registry: {
    name: string
    versions: RegistryVersion[]
    production: string | null
    staging: string | null
  }
  experiments: ExperimentRun[]
  validation: MetricSlice
  test: MetricSlice
  quality: {
    passed?: boolean
    deepchecks?: {
      available?: boolean
      passed?: boolean
      integrity_passed?: boolean
      train_test_passed?: boolean
      integrity_not_passed?: string[]
      train_test_not_passed?: string[]
    }
  }
  drift: {
    drift_detected?: boolean
    drifted_features?: string[]
    share_threshold?: number
    evidently?: { available?: boolean; wrote_html?: boolean }
  }
  deployment: { version?: string; promoted?: boolean; production?: string }
  retraining: {
    run_id?: string
    run_name?: string
    gate?: {
      passed?: boolean
      used_test_labels?: boolean
      min_validation_average_precision?: number
      min_validation_recall?: number
      validation_average_precision?: number
      validation_recall?: number
    }
  }
  investigation: {
    recomputed_on_request?: boolean
    source?: string
    measured_on?: string
    peak_horizontal_g_median_train?: number
    peak_horizontal_g_median_validation?: number
    correlation_note?: string
    extreme_event?: {
      event_id: number
      split: string
      label_name: string
      peak_horizontal_g: number
      note: string
    }
  }
  cloud: { enabled?: boolean; missing?: string[]; apis_called?: boolean }
}

export type FeatureField = {
  name: string
  label: string
  unit: string
  group: string
}

export type ExampleRow = {
  title: string
  split: string
  label: number
  event_id: number
  features: Record<string, number | null>
}

export type PredictionResponse = {
  prediction: "crash" | "non_crash"
  crash: number
  probability: number
  threshold: number
  model_name: string
  model_version: string
  feature_names: string[]
  latency_ms: number
}

export type HealthPayload = {
  checked_at: string
  fastapi: { ok: boolean }
  mlflow: { ok: boolean; detail?: string; production?: string | null }
  sqlite: { ok: boolean; path?: string; detail?: string; sensor_events?: number }
  reports: { ok: boolean; drift_html?: boolean; drift_bytes?: number | null; quality_json?: boolean; detail?: string }
  model: { loaded: boolean; version?: string; threshold?: number; detail?: string }
  docker: { operational: boolean; checked: boolean; detail: string }
  cloud: { enabled?: boolean; missing?: string[]; apis_called?: boolean; note?: string }
  predictions_recorded: boolean
}
