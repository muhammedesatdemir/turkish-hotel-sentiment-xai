export type Sentiment = "negatif" | "nötr" | "pozitif";

export interface PredictResponse {
  sentiment: Sentiment;
  confidence: number;
  probabilities: Record<string, number>;
  wc: number;
  inference_time_ms: number;
}

export interface ExplainResponse {
  sentiment: Sentiment;
  confidence: number;
  probabilities: Record<string, number>;
  lime_features: Record<string, [string, number][]>;
  tokens: string[];
  num_samples: number;
  explain_time_ms: number;
}

export interface BatchItem {
  index: number;
  text: string;
  sentiment: Sentiment;
  confidence: number;
  probabilities: Record<string, number>;
  wc: number;
}

export interface BatchResponse {
  items: BatchItem[];
  total: number;
  inference_time_ms: number;
  distribution: Record<string, number>;
}

export interface MeanStd {
  mean: number;
  std: number | null;
}

export interface OverviewMetrics {
  test_accuracy: MeanStd;
  macro_f1: MeanStd;
  weighted_f1: MeanStd;
  f1_negatif: MeanStd;
  f1_notr: MeanStd;
  f1_pozitif: MeanStd;
  test_size_per_seed: number;
  test_size_combined_3seed: number;
  dataset_size: number;
  model: string;
  chosen_seed: number;
  labels: Sentiment[];
}

export interface ConfusionMatrixData {
  labels: Sentiment[];
  matrix: number[][];
  support: Record<string, number>;
  total: number;
  note: string;
}

export interface PerClassMetrics {
  labels: Sentiment[];
  precision: Record<string, number>;
  recall: Record<string, number>;
  f1: Record<string, number>;
  support: Record<string, number>;
}

export interface MultiModelEntry {
  key: string;
  name: string;
  highlighted: boolean;
  metrics: Record<string, MeanStd>;
}

export interface MultiModelData {
  metric_keys: string[];
  models: MultiModelEntry[];
}

export interface AblationStage {
  key: string;
  label: string;
  description: string;
  metrics: Record<string, MeanStd>;
  n_seeds: number;
  highlighted?: boolean;
  note?: string;
}

export interface AblationData {
  metric_keys: string[];
  stages: AblationStage[];
}

export interface DatasetDistribution {
  total: number;
  by_class: Record<string, number>;
  by_source: Record<string, number>;
  source_descriptions?: Record<string, string>;
  splits: { train: number; val: number; test: number };
}

export interface ErrorSummary {
  total_test: number;
  total_errors: number;
  error_rate: number;
  errors_by_direction: Record<string, number>;
  errors_by_category: Record<string, number>;
  confidence_stats: {
    mean_max_prob: number;
    median_max_prob: number;
    high_conf_errors_count: number;
    low_conf_errors_count: number;
  };
  wc_stats: { mean: number; median: number; min: number; max: number };
}

export interface ExampleItem {
  id: number;
  kind: string;
  label: string;
  text: string;
  true: Sentiment;
  pred: Sentiment;
  correct: boolean;
  wc: number;
}

export interface ErrorRow {
  id: number;
  text: string;
  true: Sentiment;
  pred: Sentiment;
  direction: string;
  wc: number;
  probabilities: Record<string, number>;
  max_prob: number;
  second_prob: number;
  category: string;
  source: string;
}

export interface ErrorListResponse {
  total: number;
  limit: number;
  offset: number;
  rows: ErrorRow[];
}

export interface HealthResponse {
  status: "ok" | "loading" | "error";
  model_loaded: boolean;
  model_dir: string;
  device: string;
  labels: Sentiment[];
  error_rows: number;
}

// =============================================================================
// Yeni istatistikler (5-fold CV, McNemar, klasik baseline, sarkazm, sızıntı, EDA)
// =============================================================================

export interface CvModelEntry {
  key: string;
  name: string;
  elapsed_hours: number;
  metrics: {
    accuracy: { mean: number; std: number };
    macro_f1: { mean: number; std: number };
    weighted_f1: { mean: number; std: number };
    f1_negatif: { mean: number; std: number };
    f1_notr: { mean: number; std: number };
    f1_pozitif: { mean: number; std: number };
  };
  highlighted: boolean;
}

export interface CvResults {
  protocol: string;
  n_runs_per_model: number;
  n_folds: number;
  n_seeds: number;
  test_size_per_fold: number;
  models: CvModelEntry[];
  comparison: {
    delta_macro_f1: number;
    delta_f1_notr: number;
    speed_ratio: number;
    interpretation: string;
  };
  vs_multi_seed: { note: string };
}

export interface McnemarPairResult {
  min_p_value: number;
  median_p_value: number;
  max_p_value?: number;
  bonferroni_m3_threshold?: number;
  bonferroni_m18_threshold?: number;
  significant_m3: boolean;
  significant_m18: boolean;
  interpretation: string;
}

export interface McnemarSummary {
  total_tests: number;
  bert_electra_tests: number;
  multi_model: {
    n_tests: number;
    n_pairs: number;
    n_seeds_per_pair: number;
    bert_vs_electra: McnemarPairResult;
    bert_vs_mbert: McnemarPairResult;
    bert_vs_xlmr: McnemarPairResult;
  };
  cv_fold: {
    n_tests: number;
    comparison: string;
    min_p_value: number;
    median_p_value: number;
    max_p_value: number;
    n_significant_uncorrected: number;
    n_significant_bonferroni: number;
    fishers_combined_p: number;
    interpretation: string;
  };
  akademik_sonuc: string;
}

export interface ClassicalModelEntry {
  key: string;
  name: string;
  metrics: {
    macro_f1: { mean: number; std: number };
    f1_notr: { mean: number; std: number };
    accuracy: { mean: number; std: number };
  };
  best_classical?: boolean;
}

export interface ClassicalBaseline {
  n_seeds: number;
  n_models: number;
  n_total_runs: number;
  models: ClassicalModelEntry[];
  vs_berturk: {
    berturk_macro_f1: number;
    best_classical_macro_f1: number;
    delta_macro_f1: number;
    delta_f1_notr: number;
    interpretation: string;
  };
  top_features_per_class: {
    negatif: [string, number][];
    "nötr": [string, number][];
    pozitif: [string, number][];
  };
  xai_cross_validation: string;
}

export interface SarcasmExample {
  id: number;
  text_short: string;
  true: string;
  pred: string;
  confidence: number;
  ironic_signal: string;
}

export interface SarcasmAnalysis {
  n_total: number;
  n_sarcastic: number;
  n_control_direct_negative: number;
  n_correct: number;
  n_sarcastic_correct: number;
  n_sarcastic_wrong: number;
  control_correct: boolean;
  avg_confidence_when_wrong: number;
  interpretation: string;
  examples: SarcasmExample[];
}

export interface LeakageSeedEntry {
  seed: number;
  exact_tt: number;
  normalized_tt: number;
  jaccard_080_tt: number;
  jaccard_090_tt: number;
}

export interface LeakageAudit {
  method: string;
  multi_seed: LeakageSeedEntry[];
  multi_seed_aggregate: {
    exact_invariant_zero: boolean;
    j090_mean: number;
    j090_std: number;
  };
  cv_15_fold: {
    n_folds: number;
    exact_invariant_zero: boolean;
    normalized_mean: number;
    normalized_std: number;
    j090_mean: number;
    j090_std: number;
    j090_min: number;
    j090_max: number;
  };
  impact_bound: {
    multi_seed_test_size: number;
    cv_test_size: number;
    max_impact_macro_f1_multi_seed: number;
    max_impact_macro_f1_cv: number;
    interpretation: string;
  };
}

export interface EdaImage {
  url: string;
  title: string;
  description: string;
}

export interface EdaSummary {
  n_meta_features: number;
  key_correlations: Record<
    string,
    { pearson_r?: number; point_biserial_r?: number; interpretation: string }
  >;
  class_distribution: Record<
    string,
    { count: number; ratio?: number; wc_median?: number } | number
  >;
  source_distribution: Record<
    string,
    { count: number; ratio: number; type: string }
  >;
  images?: EdaImage[];
}
