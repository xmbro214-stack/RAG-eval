import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

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
  const [fileInputKey, setFileInputKey] = useState(0);
  const [question, setQuestion] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [savingQa, setSavingQa] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const previewRequestIdRef = useRef(0);
  const selectedDatasetPathRef = useRef<string | null>(null);

  const loadPreview = useCallback(async (dataset: DatasetItem) => {
    const requestId = previewRequestIdRef.current + 1;
    previewRequestIdRef.current = requestId;
    const requestedPath = dataset.path ?? null;
    const isLatestRequest = () =>
      previewRequestIdRef.current === requestId && selectedDatasetPathRef.current === requestedPath;

    if (!isLatestRequest()) {
      return;
    }

    setPreview(null);

    if (!requestedPath) {
      setLoadingPreview(false);
      setError("");
      return;
    }

    setLoadingPreview(true);
    setError("");
    try {
      const payload = await apiClient.previewDataset(requestedPath);
      if (isLatestRequest()) {
        setPreview(payload);
      }
    } catch (exc) {
      if (isLatestRequest()) {
        setError((exc as Error).message);
        setPreview(null);
      }
    } finally {
      if (isLatestRequest()) {
        setLoadingPreview(false);
      }
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

  useEffect(() => {
    selectedDatasetPathRef.current = selectedDataset?.path ?? null;
  }, [selectedDataset?.path]);

  async function selectDataset(dataset: DatasetItem) {
    selectedDatasetPathRef.current = dataset.path ?? null;
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
      const uploadPayload = await apiClient.uploadDataset(uploadName, uploadFile);
      const listPayload = await apiClient.listDatasets();
      const uploadedDataset =
        listPayload.datasets.find(
          (dataset) => dataset.path === uploadPayload.path && dataset.name === uploadPayload.dataset_name
        ) ??
        listPayload.datasets.find(
          (dataset) => dataset.name === uploadPayload.dataset_name || dataset.path === uploadPayload.path
        ) ??
        ({
          name: uploadPayload.dataset_name,
          path: uploadPayload.path,
          dataset_id: "",
          rows: uploadPayload.rows,
          source: "uploaded"
        } satisfies DatasetItem);

      setDatasets(listPayload.datasets);
      selectedDatasetPathRef.current = uploadedDataset.path ?? null;
      setSelectedDataset(uploadedDataset);
      setUploadName("");
      setUploadFile(null);
      setFileInputKey((current) => current + 1);
      setNotice(`Uploaded ${uploadPayload.dataset_name}`);
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
    const savedDataset = selectedDataset;
    try {
      const payload = await apiClient.saveManualQa(savedDataset.path, question.trim(), expectedAnswer.trim());
      setQuestion("");
      setExpectedAnswer("");
      setNotice(`Saved ${payload.appended_query_id}`);
      await loadDatasets();
      if (selectedDatasetPathRef.current === savedDataset.path) {
        await loadPreview(savedDataset);
      }
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setSavingQa(false);
    }
  }

  const canSaveQa = Boolean(selectedDataset?.path && question.trim() && expectedAnswer.trim() && !savingQa);
  const selectedDatasetIdentity = selectedDataset ? datasetIdentity(selectedDataset) : null;

  return (
    <section className="section-panel">
      <h1>Datasets</h1>
      {error ? <p className="error-text">{error}</p> : null}
      {notice ? <p className="notice-text">{notice}</p> : null}

      <form className="toolbar-row" onSubmit={handleUpload}>
        <input
          aria-label="Dataset name"
          disabled={uploading}
          onChange={(event) => setUploadName(event.target.value)}
          value={uploadName}
        />
        <input
          aria-label="Dataset file"
          accept=".csv,.pdf,text/csv,application/pdf"
          disabled={uploading}
          key={fileInputKey}
          onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
          type="file"
        />
        <button disabled={uploading} type="submit">
          {uploading ? "Uploading..." : "Upload dataset"}
        </button>
      </form>

      <div className="datasets-layout">
        <div aria-busy={loadingDatasets} className="data-table">
          {datasets.map((dataset) => (
            <button
              aria-label={`${dataset.name} ${dataset.path} ${dataset.rows ?? "-"} rows`}
              className={
                selectedDatasetIdentity === datasetIdentity(dataset)
                  ? "data-row button-row active"
                  : "data-row button-row"
              }
              key={datasetIdentity(dataset)}
              onClick={() => void selectDataset(dataset)}
              type="button"
            >
              <strong>{dataset.name}</strong>
              <span>{dataset.path}</span>
              <span>Rows: {dataset.rows ?? "-"}</span>
            </button>
          ))}
          {!loadingDatasets && datasets.length === 0 ? <p>No datasets found.</p> : null}
        </div>

        <div className="dataset-detail-panel">
          <div aria-busy={loadingPreview} className="dataset-preview-panel">
            <h2>{selectedDataset ? selectedDataset.name : "Preview"}</h2>
            {selectedDataset?.path ? (
              <p className="muted-text">{selectedDataset.path}</p>
            ) : (
              <p className="muted-text">Select a dataset to preview rows.</p>
            )}
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
              <textarea
                aria-label="Expected answer"
                onChange={(event) => setExpectedAnswer(event.target.value)}
                value={expectedAnswer}
              />
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
