import type {
  DatasetItem,
  DatasetPreview,
  DatasetUploadResponse,
  EvalRunItem,
  ManualQaSaveResponse,
  PipelineJob,
  ReportItem,
  RunOptions
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new ApiError(payload.error || response.statusText || "Request failed", response.status || 500);
  }
  return payload as T;
}

export const apiClient = {
  async listReports(): Promise<ReportItem[]> {
    const payload = await requestJson<{ ok: true; reports: ReportItem[] }>("/api/reports");
    return payload.reports;
  },

  async listEvalRuns(): Promise<EvalRunItem[]> {
    const payload = await requestJson<{ ok: true; runs: EvalRunItem[] }>("/api/eval-runs");
    return payload.runs;
  },

  async listDatasets(): Promise<{ datasets: DatasetItem[]; runOptions: RunOptions }> {
    const payload = await requestJson<{ ok: true; datasets: DatasetItem[]; run_options: RunOptions }>("/api/datasets");
    return { datasets: payload.datasets, runOptions: payload.run_options };
  },

  async previewDataset(path: string): Promise<DatasetPreview> {
    return requestJson<DatasetPreview>(`/api/datasets/preview?path=${encodeURIComponent(path)}`);
  },

  async getPipelineStatus(runId: string): Promise<PipelineJob> {
    const payload = await requestJson<{ ok: true; job: PipelineJob }>(
      `/api/pipeline/status?run_id=${encodeURIComponent(runId)}`
    );
    return payload.job;
  },

  async runPipeline(mode: "quick" | "standard"): Promise<{ run_id: string; status: string }> {
    return requestJson<{ ok: true; run_id: string; status: string }>("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, stage: "all" })
    });
  },

  async uploadDataset(name: string, file: File): Promise<DatasetUploadResponse> {
    const body = new FormData();
    body.set("name", name);
    body.set("file", file);
    return requestJson<DatasetUploadResponse>("/api/datasets/upload", { method: "POST", body });
  },

  async saveManualQa(
    targetDataset: string,
    question: string,
    expectedAnswer: string
  ): Promise<ManualQaSaveResponse> {
    return requestJson<ManualQaSaveResponse>("/api/datasets/manual-qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_dataset: targetDataset,
        question,
        expected_answer: expectedAnswer
      })
    });
  }
};
