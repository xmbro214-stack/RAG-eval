export type ReportItem = {
  name: string;
  path: string;
  url: string;
  modified: string;
};

export type DatasetItem = {
  name: string;
  path: string;
  dataset_id?: string;
  modified?: string;
  runnable?: boolean;
  rows?: number;
  source?: string;
};

export type RunOptions = {
  page_sizes: number[];
  similarity_thresholds: number[];
};

export type EvalRunItem = {
  run_name?: string;
  dataset?: string;
  page_size?: number;
  similarity_threshold?: number;
  report_url?: string;
  [key: string]: unknown;
};

export type JobProgress = {
  total_tasks?: number;
  completed_tasks?: number;
  current_task?: string;
  percent?: number;
};

export type PipelineJob = {
  run_id: string;
  status: string;
  progress?: JobProgress;
  log_tail?: string[];
  reports?: ReportItem[];
  error?: string;
};

export type DatasetPreview = {
  ok: true;
  rows: number;
  preview_rows: Array<Record<string, string>>;
};

export type DatasetUploadResponse = {
  ok: true;
  dataset_name: string;
  path: string;
  rows?: number;
};

export type ManualQaSaveResponse = {
  ok: true;
  appended_query_id: string;
  path?: string;
};
