export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ModelManifest {
  model_id: string;
  model_version: string;
  algorithm: string;
  stage: "experimental" | "candidate" | "production" | "rejected" | "archived";
  dataset_version: string;
  feature_version: string;
  threshold: number;
  parameters: Record<string, any>;
  metrics: {
    pr_auc?: number;
    recall?: number;
    precision?: number;
    f1?: number;
    false_positive_rate?: number;
    fp_per_10k?: number;
    confusion_matrix?: { tp: number; fp: number; tn: number; fn: number };
  };
  validation_metrics: {
    pr_auc?: number;
    recall?: number;
    precision?: number;
    f1?: number;
    false_positive_rate?: number;
    fp_per_10k?: number;
  };
  created_at: string;
  promoted_at?: string;
  artifact_path: string;
  description?: string;
}

export interface PredictResponse {
  modelVersion: string;
  algorithm: string;
  anomalyScore: number;
  isAnomaly: boolean;
  threshold: number;
  riskBand: "NORMAL" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  contributingSignals: Array<{
    feature: string;
    observedValue: number;
    baselineValue: number;
    deviation: number;
    contributionWeight: number;
    message: string;
  }>;
  featureValues: Record<string, number>;
  latencyMs: number;
}

export interface DriftReport {
  report_id: string;
  timestamp: string;
  overall_status: "NO_DRIFT" | "WARNING" | "DRIFT_DETECTED";
  reference_records: number;
  current_records: number;
  features_evaluated: number;
  drifting_features_count: number;
  warning_features_count: number;
  features: Array<{
    feature: string;
    status: string;
    psi: number;
    ks_statistic: number;
    ks_pvalue: number;
    jensen_shannon_distance: number;
    reference_mean: number;
    current_mean: number;
    reference_std: number;
    current_std: number;
  }>;
  score_drift?: {
    status: string;
    psi: number;
    ks_statistic: number;
    ks_pvalue: number;
    reference_mean_score: number;
    current_mean_score: number;
  };
}

export interface EvaluationReport {
  model_id: string;
  model_version: string;
  algorithm: string;
  metrics: {
    pr_auc: number;
    recall: number;
    precision: number;
    f1: number;
    false_positive_rate: number;
    fp_per_10k: number;
    confusion_matrix: { tp: number; fp: number; tn: number; fn: number };
  };
  threshold_tuning: {
    best_threshold: number;
    strategy: string;
    precision: number;
    recall: number;
    f1: number;
    fpr: number;
    fp_per_10k: number;
    expected_cost: number;
  };
  curves: {
    pr_curve: Array<{ x: number; y: number }>;
    roc_curve: Array<{ x: number; y: number }>;
    score_distribution: Array<{
      bin_start: number;
      bin_end: number;
      normal_density: number;
      anomaly_density: number;
    }>;
  };
  error_analysis: {
    total_false_positives: number;
    total_false_negatives: number;
    top_false_positives: Array<any>;
    top_false_negatives: Array<any>;
    fp_categories: Record<string, number>;
    fn_categories: Record<string, number>;
  };
}

export const api = {
  async getHealth() {
    const res = await fetch(`${API_BASE_URL}/health/ready`);
    return res.json();
  },

  async getModels(): Promise<ModelManifest[]> {
    const res = await fetch(`${API_BASE_URL}/api/v1/models`);
    return res.json();
  },

  async getCurrentModel(): Promise<ModelManifest> {
    const res = await fetch(`${API_BASE_URL}/api/v1/models/current`);
    return res.json();
  },

  async promoteModel(modelVersion: string, stage: string) {
    const res = await fetch(`${API_BASE_URL}/api/v1/models/${modelVersion}/promote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage }),
    });
    return res.json();
  },

  async predict(payload: any): Promise<PredictResponse> {
    const res = await fetch(`${API_BASE_URL}/api/v1/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  },

  async predictBatch(events: any[]) {
    const res = await fetch(`${API_BASE_URL}/api/v1/predict/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events }),
    });
    return res.json();
  },

  async getEvaluations(): Promise<EvaluationReport[]> {
    const res = await fetch(`${API_BASE_URL}/api/v1/evaluations`);
    return res.json();
  },

  async getEvaluation(modelVersion: string): Promise<EvaluationReport> {
    const res = await fetch(`${API_BASE_URL}/api/v1/evaluations/${modelVersion}`);
    return res.json();
  },

  async getDriftReport(): Promise<DriftReport> {
    const res = await fetch(`${API_BASE_URL}/api/v1/drift/latest`);
    return res.json();
  },

  async runDriftCheck(): Promise<DriftReport> {
    const res = await fetch(`${API_BASE_URL}/api/v1/drift/run`, {
      method: "POST",
    });
    return res.json();
  },

  async getDatasets() {
    const res = await fetch(`${API_BASE_URL}/api/v1/datasets`);
    return res.json();
  },

  async getFeaturesSummary() {
    const res = await fetch(`${API_BASE_URL}/api/v1/datasets/summary`);
    return res.json();
  },

  async getQualityReport() {
    const res = await fetch(`${API_BASE_URL}/api/v1/datasets/quality`);
    return res.json();
  },
};
