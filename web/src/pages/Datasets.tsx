import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiClient } from "../api/client";
import type { DatasetItem, DatasetPreview, RetrievedPassage } from "../api/types";
import type { LanguageCode, Translate } from "../i18n";

function datasetIdentity(dataset: DatasetItem) {
  return [dataset.source || "", dataset.dataset_id || "", dataset.name || "", dataset.path || ""].join("|");
}

function datasetOptionLabel(dataset: DatasetItem, rowsLabel: string) {
  return `${dataset.name} - ${dataset.path || "-"} (${dataset.rows ?? "-"} ${rowsLabel.toLowerCase()})`;
}

function isUploadedDatasetArtifact(dataset: DatasetItem) {
  return (dataset.path || "").replace(/\\/g, "/").includes("/uploaded_datasets/");
}

type DatasetsProps = {
  language: LanguageCode;
  t: Translate;
};

export function Datasets({ language, t }: DatasetsProps) {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetItem | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [question, setQuestion] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [retrievedPassages, setRetrievedPassages] = useState<RetrievedPassage[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [generatingAnswer, setGeneratingAnswer] = useState(false);
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

  const visibleDatasets = useMemo(
    () => datasets.filter((dataset) => !isUploadedDatasetArtifact(dataset)),
    [datasets]
  );

  useEffect(() => {
    if (selectedDataset || visibleDatasets.length === 0) {
      return;
    }

    const firstDataset = visibleDatasets[0];
    selectedDatasetPathRef.current = firstDataset.path ?? null;
    setSelectedDataset(firstDataset);
    void loadPreview(firstDataset);
  }, [loadPreview, selectedDataset, visibleDatasets]);

  async function selectDataset(dataset: DatasetItem) {
    selectedDatasetPathRef.current = dataset.path ?? null;
    setSelectedDataset(dataset);
    setNotice("");
    setRetrievedPassages([]);
    await loadPreview(dataset);
  }

  async function handleDatasetSelection(datasetKey: string) {
    const dataset = visibleDatasets.find((item) => datasetIdentity(item) === datasetKey);
    if (dataset) {
      await selectDataset(dataset);
    }
  }

  async function handleUploadFile(file: File | null) {
    setNotice("");
    setError("");

    if (!file) {
      setError(t("chooseDatasetFile"));
      return;
    }

    setUploading(true);
    try {
      const uploadPayload = await apiClient.uploadDataset("", file);
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
      setFileInputKey((current) => current + 1);
      setRetrievedPassages([]);
      setNotice(`${t("uploaded")} ${uploadPayload.dataset_name}`);
      await loadPreview(uploadedDataset);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function handleGenerateAnswer() {
    setNotice("");
    setError("");

    const cleanQuestion = question.trim();
    if (!cleanQuestion) {
      setError(t("enterQuestionBeforeGenerating"));
      return;
    }

    setGeneratingAnswer(true);
    try {
      const payload = await apiClient.generateAnswer(cleanQuestion, language);
      setExpectedAnswer(payload.expected_answer || "");
      setRetrievedPassages(payload.passages || []);
      setNotice(t("generatedAnswer"));
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setGeneratingAnswer(false);
    }
  }

  async function handleSaveManualQa(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setNotice("");
    setError("");

    const targetDatasetPath = selectedDataset?.path || "";

    if (!targetDatasetPath) {
      setError(t("selectDatasetBeforeSaving"));
      return;
    }

    if (!question.trim() || !expectedAnswer.trim()) {
      setError(t("enterQuestionAndAnswer"));
      return;
    }

    setSavingQa(true);
    const selectedPreviewDataset = selectedDataset;
    try {
      const payload = await apiClient.saveManualQa(targetDatasetPath, question.trim(), expectedAnswer.trim());
      setQuestion("");
      setExpectedAnswer("");
      setRetrievedPassages([]);
      setNotice(`Saved ${payload.appended_query_id}`);
      await loadDatasets();
      if (selectedPreviewDataset && selectedDatasetPathRef.current === targetDatasetPath) {
        await loadPreview(selectedPreviewDataset);
      }
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setSavingQa(false);
    }
  }

  const canSaveQa = Boolean(selectedDataset?.path && question.trim() && expectedAnswer.trim() && !savingQa);
  const canGenerateAnswer = Boolean(question.trim() && !generatingAnswer);
  return (
    <section className="section-panel datasets-page">
      {error ? <p className="error-text">{error}</p> : null}
      {notice ? <p className="notice-text">{notice}</p> : null}

      <div className="datasets-layout datasets-d2-layout">
        <div className="dataset-control-bar dataset-console-card surface-panel" aria-busy={loadingDatasets}>
          <label>
            <span>Use dataset</span>
            <select
              aria-label="Use dataset"
              className="dataset-select"
              disabled={loadingDatasets || visibleDatasets.length === 0}
              onChange={(event) => void handleDatasetSelection(event.target.value)}
              value={selectedDataset ? datasetIdentity(selectedDataset) : ""}
            >
              {visibleDatasets.map((dataset) => (
                <option key={datasetIdentity(dataset)} value={datasetIdentity(dataset)}>
                  {datasetOptionLabel(dataset, t("rows"))}
                </option>
              ))}
            </select>
          </label>
          <label className={uploading ? "file-picker-button upload-file-action disabled" : "file-picker-button upload-file-action"}>
            <span>{uploading ? t("uploading") : t("uploadDataset")}</span>
            <input
              aria-label="Dataset file"
              accept=".csv,.pdf,text/csv,application/pdf"
              className="hidden-file-input"
              disabled={uploading}
              key={fileInputKey}
              onChange={(event) => void handleUploadFile(event.target.files?.[0] ?? null)}
              type="file"
            />
          </label>
          {!loadingDatasets && visibleDatasets.length === 0 ? <p>{t("noDatasetsFound")}</p> : null}
        </div>

        <div className="dataset-workspace-panel">
          <form className="manual-qa-form-panel manual-qa-card primary-qa-panel surface-panel" onSubmit={handleSaveManualQa}>
            <h2>{t("manualQa")}</h2>
            <div className="manual-qa-primary-grid manual-qa-symmetric-grid">
              <div className="manual-qa-column question-column">
                <label className="qa-main-field">
                  <span>{t("question")}</span>
                  <textarea
                    aria-label="Question"
                    className="qa-main-textarea"
                    onChange={(event) => {
                      setQuestion(event.target.value);
                      setRetrievedPassages([]);
                    }}
                    value={question}
                  />
                </label>
                <div className="qa-field-action-row manual-qa-generation-actions">
                  <button
                    className="manual-qa-secondary-action manual-qa-generate-action"
                    disabled={!canGenerateAnswer}
                    onClick={() => void handleGenerateAnswer()}
                    type="button"
                  >
                    {generatingAnswer ? t("generating") : t("generateAnswer")}
                  </button>
                </div>
              </div>
              <div className="manual-qa-column-divider" aria-hidden="true" />
              <div className="manual-qa-column answer-column">
                <label className="qa-main-field">
                  <span>{t("expectedAnswer")}</span>
                  <textarea
                    aria-label="Expected answer"
                    className="qa-main-textarea"
                    onChange={(event) => setExpectedAnswer(event.target.value)}
                    value={expectedAnswer}
                  />
                </label>
                <div className="qa-field-action-row manual-qa-save-actions">
                  <button className="manual-qa-save-action" disabled={!canSaveQa} type="submit">
                    {savingQa ? t("saving") : t("saveQa")}
                  </button>
                </div>
              </div>
            </div>
          </form>
          <div className="dataset-disclosure-stack">
            <details className="dataset-details-panel auxiliary-panel">
                <summary>{selectedDataset ? `${t("preview")}: ${selectedDataset.name}` : t("preview")}</summary>
                <div aria-busy={loadingPreview} className="dataset-preview-panel">
                  <h2>{selectedDataset ? selectedDataset.name : t("preview")}</h2>
                  {selectedDataset?.path ? (
                    <p className="muted-text">{selectedDataset.path}</p>
                  ) : (
                    <p className="muted-text">{t("selectDatasetPreview")}</p>
                  )}
                  {preview ? (
                    <p>
                      {preview.rows} {t("rows")}
                    </p>
                  ) : null}
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
                  {preview && preview.preview_rows.length === 0 ? <p>{t("noPreviewRows")}</p> : null}
                </div>
              </details>
              <details className="retrieved-passages-panel auxiliary-panel" open={retrievedPassages.length > 0}>
                <summary>{t("retrievedPassages")}</summary>
                {retrievedPassages.length > 0 ? (
                  <div className="retrieved-passages-list">
                    {retrievedPassages.map((passage, index) => (
                      <article className="retrieved-passage-item" key={`${passage.id || passage.source || "passage"}-${index}`}>
                        <strong>{passage.id || passage.source || `passage_${index + 1}`}</strong>
                        <p>{String(passage.text || passage.content || "")}</p>
                        {passage.score !== undefined || passage.source ? (
                          <span>
                            {passage.source ? String(passage.source) : ""}
                            {passage.score !== undefined ? ` score ${Number(passage.score).toFixed(3)}` : ""}
                          </span>
                        ) : null}
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="muted-text">{t("noRetrievedPassages")}</p>
                )}
              </details>
          </div>
        </div>
      </div>
    </section>
  );
}
