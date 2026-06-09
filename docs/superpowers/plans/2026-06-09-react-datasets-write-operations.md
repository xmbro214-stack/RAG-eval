# React Datasets Write Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the React `Datasets` page to upload, preview, and manual Q&A save backend routes.

**Architecture:** Keep the workflow inside `web/src/pages/Datasets.tsx`, with `apiClient` exposing one new manual-save method. The page owns request state, validates forms before writes, refreshes inventory after upload, and reloads preview after dataset selection or manual Q&A save.

**Tech Stack:** React, TypeScript, Vite, Vitest, Testing Library, FastAPI backend routes already implemented.

---

## User Preference

- Do not push to any remote.
- Do not create intermediate local commits while implementing this feature.
- After all implementation work and verification pass, create one local commit for the complete feature.

## File Structure

- Modify `web/src/api/types.ts`: add small response types for dataset upload and manual Q&A save.
- Modify `web/src/api/client.ts`: add `saveManualQa()` and narrow `uploadDataset()` return type.
- Modify `web/src/api/client.test.ts`: test manual Q&A request shape and retained shared error handling.
- Modify `web/src/pages/Datasets.tsx`: connect upload, selection, preview, and manual Q&A save.
- Modify `web/src/App.test.tsx`: cover user-visible dataset write workflow.
- Modify `web/src/styles/globals.css`: add compact form, selected-row, preview, and notice styles.

---

### Task 1: Add API Client Contract

**Files:**
- Modify: `web/src/api/types.ts`
- Modify: `web/src/api/client.ts`
- Modify: `web/src/api/client.test.ts`

- [ ] **Step 1: Add failing API client tests**

Replace `web/src/api/client.test.ts` with:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";

import { apiClient, ApiError } from "./client";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("apiClient", () => {
  it("loads reports", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ ok: true, reports: [{ name: "smoke.html", url: "/reports/smoke.html" }] })
      })
    );

    const reports = await apiClient.listReports();

    expect(reports[0].name).toBe("smoke.html");
  });

  it("saves manual Q&A rows with backend field names", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ ok: true, appended_query_id: "query_2" })
    });
    vi.stubGlobal("fetch", fetchMock);

    const payload = await apiClient.saveManualQa("data/custom.csv", "What is A?", "Answer A");

    expect(payload.appended_query_id).toBe("query_2");
    expect(fetchMock).toHaveBeenCalledWith("/api/datasets/manual-qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_dataset: "data/custom.csv",
        question: "What is A?",
        expected_answer: "Answer A"
      })
    });
  });

  it("throws ApiError for backend errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({ ok: false, error: "Upload failed" })
      })
    );

    await expect(apiClient.listReports()).rejects.toEqual(new ApiError("Upload failed", 500));
  });
});
```

- [ ] **Step 2: Run API client tests and verify failure**

Run:

```powershell
cd web
npm test -- src/api/client.test.ts
```

Expected: FAIL because `apiClient.saveManualQa` does not exist.

- [ ] **Step 3: Add dataset write response types**

Append these types to `web/src/api/types.ts`:

```ts
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
```

- [ ] **Step 4: Implement API client methods**

Update the import in `web/src/api/client.ts` to:

```ts
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
```

Replace the existing `uploadDataset` method and add `saveManualQa` below it:

```ts
  async uploadDataset(name: string, file: File): Promise<DatasetUploadResponse> {
    const body = new FormData();
    body.set("name", name);
    body.set("file", file);
    return requestJson<DatasetUploadResponse>("/api/datasets/upload", { method: "POST", body });
  },

  async saveManualQa(targetDataset: string, question: string, expectedAnswer: string): Promise<ManualQaSaveResponse> {
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
```

- [ ] **Step 5: Run API client tests and verify pass**

Run:

```powershell
cd web
npm test -- src/api/client.test.ts
```

Expected: PASS, 3 tests passing.

---

### Task 2: Add Dataset Page Workflow Tests

**Files:**
- Modify: `web/src/App.test.tsx`

- [ ] **Step 1: Replace App tests with dataset workflow coverage**

Replace `web/src/App.test.tsx` with:

```tsx
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

function makeJsonResponse(payload: object) {
  return {
    ok: true,
    json: async () => payload
  };
}

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);

      if (path === "/api/reports") {
        return makeJsonResponse({
          ok: true,
          reports: [
            {
              name: "smoke.html",
              path: "reports/smoke.html",
              url: "/reports/smoke.html",
              modified: "2026-06-09"
            }
          ]
        });
      }

      if (path === "/api/eval-runs") {
        return makeJsonResponse({
          ok: true,
          runs: [{ run_name: "smoke", dataset: "qa", report_url: "/reports/smoke.html" }]
        });
      }

      if (path === "/api/datasets") {
        return makeJsonResponse({
          ok: true,
          datasets: [
            { name: "custom", path: "data/custom.csv", rows: 1 },
            { name: "support_qa", path: "data/support_qa.csv", rows: 2 }
          ],
          run_options: { page_sizes: [5], similarity_thresholds: [0.1] }
        });
      }

      if (path === "/api/datasets/preview?path=data%2Fcustom.csv") {
        return makeJsonResponse({
          ok: true,
          rows: 1,
          preview_rows: [{ query_id: "q1", query: "What is A?", expected_answer: "Answer A" }]
        });
      }

      if (path === "/api/datasets/preview?path=data%2Fsupport_qa.csv") {
        return makeJsonResponse({
          ok: true,
          rows: 2,
          preview_rows: [{ query_id: "q2", query: "What is B?", expected_answer: "Answer B" }]
        });
      }

      if (path === "/api/datasets/upload" && init?.method === "POST") {
        return makeJsonResponse({
          ok: true,
          dataset_name: "support_qa",
          path: "data/support_qa.csv",
          rows: 2
        });
      }

      if (path === "/api/datasets/manual-qa" && init?.method === "POST") {
        return makeJsonResponse({ ok: true, appended_query_id: "query_2" });
      }

      return makeJsonResponse({ ok: true });
    })
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("App", () => {
  it("loads datasets on the default page", async () => {
    render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
  });

  it("loads dataset preview when selecting a dataset", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: /custom data\/custom\.csv 1 rows/i }));

    expect(await screen.findByText("What is A?")).toBeInTheDocument();
    expect(screen.getByText("1 rows")).toBeInTheDocument();
  });

  it("uploads a dataset and refreshes inventory", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText("Dataset name"), "Support QA");
    await user.upload(
      screen.getByLabelText("Dataset file"),
      new File(["query_id,query,expected_answer\nq1,What is A?,Answer A\n"], "qa.csv", { type: "text/csv" })
    );
    await user.click(screen.getByRole("button", { name: "Upload dataset" }));

    expect(await screen.findByText("Uploaded support_qa")).toBeInTheDocument();
    expect(screen.getAllByText("support_qa").length).toBeGreaterThan(0);
  });

  it("saves manual Q&A to the selected dataset", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: /custom data\/custom\.csv 1 rows/i }));
    await user.type(screen.getByLabelText("Question"), "What is B?");
    await user.type(screen.getByLabelText("Expected answer"), "Answer B");
    await user.click(screen.getByRole("button", { name: "Save Q&A" }));

    expect(await screen.findByText("Saved query_2")).toBeInTheDocument();
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock).toHaveBeenCalledWith("/api/datasets/manual-qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_dataset: "data/custom.csv",
        question: "What is B?",
        expected_answer: "Answer B"
      })
    });
  });

  it("loads reports when switching to Reports", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Reports" }));

    expect(await screen.findByText("smoke.html")).toBeInTheDocument();
  });

  it("loads evaluation runs when switching to Records", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Records" }));

    expect(await screen.findByText("smoke")).toBeInTheDocument();
  });

  it("shows run controls", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Standard evaluation" })).toBeInTheDocument());
  });
});
```

- [ ] **Step 2: Run App tests and verify failure**

Run:

```powershell
cd web
npm test -- src/App.test.tsx
```

Expected: FAIL because the `Datasets` page does not yet render selectable dataset buttons, preview rows, active upload behavior, or the manual Q&A form.

---

### Task 3: Implement Datasets Page Workflow

**Files:**
- Modify: `web/src/pages/Datasets.tsx`
- Modify: `web/src/styles/globals.css`

- [ ] **Step 1: Replace `Datasets.tsx` with connected workflow**

Replace `web/src/pages/Datasets.tsx` with:

```tsx
import { FormEvent, useCallback, useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { DatasetItem, DatasetPreview } from "../api/types";

function datasetIdentity(dataset: DatasetItem) {
  return [dataset.source || "", dataset.dataset_id || "", dataset.name || "", dataset.path || ""].join("|");
}

export function Datasets() {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetItem | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [savingQa, setSavingQa] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadPreview = useCallback(async (dataset: DatasetItem) => {
    if (!dataset.path) {
      setPreview(null);
      return;
    }
    setLoadingPreview(true);
    setError("");
    try {
      const payload = await apiClient.previewDataset(dataset.path);
      setPreview(payload);
    } catch (exc) {
      setError((exc as Error).message);
      setPreview(null);
    } finally {
      setLoadingPreview(false);
    }
  }, []);

  const loadDatasets = useCallback(async () => {
    setLoadingDatasets(true);
    setError("");
    try {
      const payload = await apiClient.listDatasets();
      setDatasets(payload.datasets);
      setSelectedDataset((current) => {
        if (!current) {
          return current;
        }
        return payload.datasets.find((dataset) => datasetIdentity(dataset) === datasetIdentity(current)) ?? current;
      });
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setLoadingDatasets(false);
    }
  }, []);

  useEffect(() => {
    void loadDatasets();
  }, [loadDatasets]);

  async function selectDataset(dataset: DatasetItem) {
    setSelectedDataset(dataset);
    setNotice("");
    await loadPreview(dataset);
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setNotice("");
    setError("");
    if (!uploadFile) {
      setError("Choose a dataset file before uploading.");
      return;
    }
    setUploading(true);
    try {
      const payload = await apiClient.uploadDataset(uploadName, uploadFile);
      const listPayload = await apiClient.listDatasets();
      setDatasets(listPayload.datasets);
      const uploadedDataset =
        listPayload.datasets.find((dataset) => dataset.path === payload.path && dataset.name === payload.dataset_name) ??
        listPayload.datasets.find((dataset) => dataset.name === payload.dataset_name || dataset.path === payload.path) ??
        ({ name: payload.dataset_name, path: payload.path, rows: payload.rows } satisfies DatasetItem);
      setSelectedDataset(uploadedDataset);
      setUploadName("");
      setUploadFile(null);
      setNotice(`Uploaded ${payload.dataset_name}`);
      await loadPreview(uploadedDataset);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function handleSaveManualQa(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setNotice("");
    setError("");
    if (!selectedDataset?.path) {
      setError("Select a dataset before saving Q&A.");
      return;
    }
    if (!question.trim() || !expectedAnswer.trim()) {
      setError("Enter both a question and expected answer.");
      return;
    }
    setSavingQa(true);
    try {
      const payload = await apiClient.saveManualQa(selectedDataset.path, question.trim(), expectedAnswer.trim());
      setQuestion("");
      setExpectedAnswer("");
      setNotice(`Saved ${payload.appended_query_id}`);
      await loadPreview(selectedDataset);
      await loadDatasets();
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setSavingQa(false);
    }
  }

  const canSaveQa = Boolean(selectedDataset?.path && question.trim() && expectedAnswer.trim() && !savingQa);

  return (
    <section className="section-panel">
      <h1>Datasets</h1>
      {error ? <p className="error-text">{error}</p> : null}
      {notice ? <p className="notice-text">{notice}</p> : null}

      <form className="toolbar-row" onSubmit={handleUpload}>
        <input aria-label="Dataset name" onChange={(event) => setUploadName(event.target.value)} value={uploadName} />
        <input
          aria-label="Dataset file"
          accept=".csv,.pdf,text/csv,application/pdf"
          onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
          type="file"
        />
        <button disabled={uploading} type="submit">
          {uploading ? "Uploading..." : "Upload dataset"}
        </button>
      </form>

      <div className="datasets-layout">
        <div className="data-table" aria-busy={loadingDatasets}>
          {datasets.map((dataset) => (
            <button
              className={selectedDataset && datasetIdentity(selectedDataset) === datasetIdentity(dataset) ? "data-row button-row active" : "data-row button-row"}
              key={datasetIdentity(dataset)}
              onClick={() => void selectDataset(dataset)}
              type="button"
            >
              <strong>{dataset.name}</strong>
              <span>{dataset.path}</span>
              <span>{dataset.rows ?? "-"} rows</span>
            </button>
          ))}
          {!loadingDatasets && datasets.length === 0 ? <p>No datasets found.</p> : null}
        </div>

        <div className="dataset-detail-panel">
          <div className="dataset-preview-panel" aria-busy={loadingPreview}>
            <h2>{selectedDataset ? selectedDataset.name : "Preview"}</h2>
            {selectedDataset?.path ? <p className="muted-text">{selectedDataset.path}</p> : <p className="muted-text">Select a dataset to preview rows.</p>}
            {preview ? <p>{preview.rows} rows</p> : null}
            {preview && preview.preview_rows.length > 0 ? (
              <div className="preview-table">
                {preview.preview_rows.map((row, index) => (
                  <div className="preview-row" key={`${row.query_id || "row"}-${index}`}>
                    <strong>{row.query_id || `row_${index + 1}`}</strong>
                    <span>{row.query || "-"}</span>
                    <span>{row.expected_answer || "-"}</span>
                  </div>
                ))}
              </div>
            ) : null}
            {preview && preview.preview_rows.length === 0 ? <p>No preview rows found.</p> : null}
          </div>

          <form className="manual-qa-form-panel" onSubmit={handleSaveManualQa}>
            <h2>Manual Q&A</h2>
            <label>
              <span>Question</span>
              <textarea aria-label="Question" onChange={(event) => setQuestion(event.target.value)} value={question} />
            </label>
            <label>
              <span>Expected answer</span>
              <textarea aria-label="Expected answer" onChange={(event) => setExpectedAnswer(event.target.value)} value={expectedAnswer} />
            </label>
            <button disabled={!canSaveQa} type="submit">
              {savingQa ? "Saving..." : "Save Q&A"}
            </button>
          </form>
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Add page styles**

Append to `web/src/styles/globals.css`:

```css
.notice-text {
  color: #146c43;
  font-weight: 700;
}

.datasets-layout {
  display: grid;
  grid-template-columns: minmax(260px, 420px) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.data-row.active {
  border-color: #005bac;
  background: #eef6ff;
}

.dataset-detail-panel {
  display: grid;
  gap: 16px;
  min-width: 0;
}

.dataset-preview-panel,
.manual-qa-form-panel {
  display: grid;
  gap: 10px;
  min-width: 0;
  padding: 14px;
  border: 1px solid #d8e4f0;
  border-radius: 6px;
  background: #fff;
}

.dataset-preview-panel h2,
.manual-qa-form-panel h2 {
  margin: 0;
  color: #172033;
  font-size: 16px;
}

.muted-text {
  margin: 0;
  color: #556575;
  overflow-wrap: anywhere;
}

.preview-table {
  display: grid;
  gap: 6px;
}

.preview-row {
  display: grid;
  grid-template-columns: minmax(72px, 120px) minmax(0, 1fr) minmax(0, 1fr);
  gap: 10px;
  padding: 8px;
  border: 1px solid #d8e4f0;
  border-radius: 6px;
  background: #f8fbfe;
}

.manual-qa-form-panel label {
  display: grid;
  gap: 6px;
}

.manual-qa-form-panel textarea {
  min-height: 86px;
  border: 1px solid #c9d7e5;
  border-radius: 6px;
  padding: 8px;
  resize: vertical;
}

.manual-qa-form-panel button {
  justify-self: start;
  min-height: 36px;
  border: 0;
  border-radius: 6px;
  padding: 0 12px;
  color: #fff;
  background: #005bac;
  cursor: pointer;
}

.manual-qa-form-panel button:disabled,
.toolbar-row button:disabled {
  cursor: not-allowed;
  opacity: .64;
}

@media (max-width: 900px) {
  .datasets-layout {
    grid-template-columns: 1fr;
  }

  .preview-row {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 3: Run App tests and verify pass**

Run:

```powershell
cd web
npm test -- src/App.test.tsx
```

Expected: PASS.

---

### Task 4: Full Verification and Local Commit

**Files:**
- Modify: `web/src/api/types.ts`
- Modify: `web/src/api/client.ts`
- Modify: `web/src/api/client.test.ts`
- Modify: `web/src/pages/Datasets.tsx`
- Modify: `web/src/App.test.tsx`
- Modify: `web/src/styles/globals.css`
- Add: `docs/superpowers/plans/2026-06-09-react-datasets-write-operations.md`

- [ ] **Step 1: Run all frontend tests**

Run:

```powershell
cd web
npm test
```

Expected: PASS, all Vitest tests pass.

- [ ] **Step 2: Run frontend production build**

Run:

```powershell
cd web
npm run build
```

Expected: PASS, Vite emits `dist/index.html` and assets.

- [ ] **Step 3: Run backend tests**

Run:

```powershell
python -m pytest -q
```

Expected: PASS, all pytest tests pass.

- [ ] **Step 4: Inspect final diff**

Run:

```powershell
git diff --stat
git diff -- web/src/api/client.ts web/src/api/types.ts web/src/pages/Datasets.tsx
```

Expected: Diff only contains the dataset write workflow, API client additions, tests, styles, and this plan.

- [ ] **Step 5: Create one local commit**

Run:

```powershell
git add web/src/api/types.ts web/src/api/client.ts web/src/api/client.test.ts web/src/pages/Datasets.tsx web/src/App.test.tsx web/src/styles/globals.css docs/superpowers/plans/2026-06-09-react-datasets-write-operations.md
git commit -m "feat: connect react dataset write workflow"
```

Expected: One local commit is created. Do not push to any remote.

---

## Self-Review Checklist

- Spec coverage: upload, select preview, and single manual Q&A save are covered by Tasks 1-3.
- Scope: bulk Q&A, AI Q&A candidates, report chat, full browsing, and broad visual redesign are excluded.
- Type consistency: `DatasetUploadResponse`, `ManualQaSaveResponse`, and `DatasetPreview` match API client usage.
- Request fields: manual Q&A posts `target_dataset`, `question`, and `expected_answer`.
- Verification: frontend tests, frontend build, backend tests, diff inspection, and one local commit are included.
