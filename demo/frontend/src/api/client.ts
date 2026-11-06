import type {
  AblationData,
  BatchResponse,
  ClassicalBaseline,
  ConfusionMatrixData,
  CvResults,
  DatasetDistribution,
  EdaSummary,
  ErrorListResponse,
  ErrorRow,
  ErrorSummary,
  ExampleItem,
  ExplainResponse,
  HealthResponse,
  LeakageAudit,
  McnemarSummary,
  MultiModelData,
  OverviewMetrics,
  PerClassMetrics,
  PredictResponse,
  SarcasmAnalysis,
} from "./types";

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text();
    }
    const msg =
      typeof detail === "object" && detail && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `HTTP ${res.status}`;
    throw new ApiError(res.status, detail, msg);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  predict: (text: string) =>
    request<PredictResponse>("/api/predict", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  explain: (text: string, num_samples = 300) =>
    request<ExplainResponse>("/api/explain", {
      method: "POST",
      body: JSON.stringify({ text, num_samples }),
    }),

  batch: (texts: string[]) =>
    request<BatchResponse>("/api/predict/batch", {
      method: "POST",
      body: JSON.stringify({ texts }),
    }),

  batchUpload: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE_URL}/api/predict/batch/upload`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      let detail: unknown = null;
      try { detail = await res.json(); } catch { detail = await res.text(); }
      const msg =
        typeof detail === "object" && detail && "detail" in detail
          ? String((detail as { detail: unknown }).detail)
          : `HTTP ${res.status}`;
      throw new ApiError(res.status, detail, msg);
    }
    return (await res.json()) as BatchResponse;
  },

  overview:        () => request<OverviewMetrics>("/api/stats/overview"),
  confusionMatrix: () => request<ConfusionMatrixData>("/api/stats/confusion-matrix"),
  perClass:        () => request<PerClassMetrics>("/api/stats/per-class"),
  multiModel:      () => request<MultiModelData>("/api/stats/multi-model"),
  ablation:        () => request<AblationData>("/api/stats/ablation"),
  dataset:         () => request<DatasetDistribution>("/api/stats/dataset"),
  errorSummary:    () => request<ErrorSummary>("/api/stats/error-summary"),
  examples:        () => request<{ items: ExampleItem[] }>("/api/examples"),

  // 2026-05-10 sertleştirme paketi: yeni 6 endpoint
  cv:                () => request<CvResults>("/api/stats/cv"),
  mcnemar:           () => request<McnemarSummary>("/api/stats/mcnemar"),
  classicalBaseline: () => request<ClassicalBaseline>("/api/stats/classical-baseline"),
  sarcasm:           () => request<SarcasmAnalysis>("/api/stats/sarcasm"),
  leakage:           () => request<LeakageAudit>("/api/stats/leakage"),
  eda:               () => request<EdaSummary>("/api/stats/eda"),

  errors: (params: {
    direction?: string;
    category?: string;
    min_conf?: number;
    max_conf?: number;
    limit?: number;
    offset?: number;
  } = {}) => {
    const usp = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") usp.set(k, String(v));
    });
    const qs = usp.toString();
    return request<ErrorListResponse>(`/api/errors${qs ? `?${qs}` : ""}`);
  },

  error: (id: number) => request<ErrorRow>(`/api/errors/${id}`),
};

export { ApiError };
