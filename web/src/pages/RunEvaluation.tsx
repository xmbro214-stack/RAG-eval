import { useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { DatasetItem, PipelineJob, RunOptions } from "../api/types";
import type { Translate } from "../i18n";

const CUSTOM_OPTION = "custom";

type RunEvaluationProps = {
  onJobStart: (job: PipelineJob) => void;
  t: Translate;
};

export function RunEvaluation({ onJobStart, t }: RunEvaluationProps) {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [runOptions, setRunOptions] = useState<RunOptions>({ page_sizes: [], similarity_thresholds: [] });
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedPageSize, setSelectedPageSize] = useState("");
  const [selectedSimilarity, setSelectedSimilarity] = useState("");
  const [customPageSize, setCustomPageSize] = useState("");
  const [customSimilarity, setCustomSimilarity] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadRunOptions() {
      setLoadingOptions(true);
      setError("");
      try {
        const payload = await apiClient.listDatasets();
        if (cancelled) return;

        const runnableDatasets = payload.datasets.filter(
          (dataset) => dataset.source === "config" && dataset.runnable !== false
        );
        setDatasets(runnableDatasets);
        setRunOptions(payload.runOptions);
        setSelectedDatasets(runnableDatasets.map((dataset) => dataset.name));
        const firstPageSize = payload.runOptions.page_sizes[0];
        const firstSimilarity = payload.runOptions.similarity_thresholds[0];
        setSelectedPageSize(String(firstPageSize ?? ""));
        setSelectedSimilarity(String(firstSimilarity ?? ""));
        setCustomPageSize(String(firstPageSize ?? ""));
        setCustomSimilarity(String(firstSimilarity ?? ""));
      } catch (exc) {
        if (!cancelled) {
          setError((exc as Error).message);
        }
      } finally {
        if (!cancelled) {
          setLoadingOptions(false);
        }
      }
    }

    void loadRunOptions();

    return () => {
      cancelled = true;
    };
  }, []);

  function toggleDataset(name: string) {
    setSelectedDatasets((current) =>
      current.includes(name) ? current.filter((dataset) => dataset !== name) : [...current, name]
    );
  }

  async function start(mode: "quick" | "standard") {
    setError("");
    setNotice("");
    const pageSizeValue = selectedPageSize === CUSTOM_OPTION ? customPageSize : selectedPageSize;
    const similarityValue = selectedSimilarity === CUSTOM_OPTION ? customSimilarity : selectedSimilarity;
    const pageSize = Number(pageSizeValue);
    const similarity = Number(similarityValue);

    if (!selectedDatasets.length) {
      setError(t("selectOneDataset"));
      return;
    }
    if (!Number.isFinite(pageSize) || pageSize <= 0 || !Number.isFinite(similarity) || similarity < 0) {
      setError(t("selectRetrievalSettings"));
      return;
    }

    setStarting(true);
    try {
      const payload = await apiClient.runPipeline(mode, {
        datasets: selectedDatasets,
        page_sizes: [pageSize],
        similarity_thresholds: [similarity]
      });
      onJobStart({ run_id: payload.run_id, status: payload.status });
      setNotice(`${t("started")} ${payload.run_id}`);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setStarting(false);
    }
  }

  const selectedDatasetSummary = selectedDatasets.length > 0 ? selectedDatasets.join(", ") : "-";
  const selectedPageSizeSummary = selectedPageSize === CUSTOM_OPTION ? customPageSize || "-" : selectedPageSize || "-";
  const selectedSimilaritySummary =
    selectedSimilarity === CUSTOM_OPTION ? customSimilarity || "-" : selectedSimilarity || "-";

  return (
    <section className="section-panel run-page">
      {error ? <p className="error-text">{error}</p> : null}
      {notice ? <p className="notice-text">{notice}</p> : null}
      <div aria-busy={loadingOptions} className="run-flow-panel run-flow-wide-panel surface-panel">
        <section className="run-flow-step datasets-step">
          <div className="run-step-label">
            <span className="run-step-number">1</span>
            <div>
              <h2>{t("runDatasets")}</h2>
              <p>{t("selectOneDataset")}</p>
            </div>
          </div>
          <div className="run-step-content run-dataset-choices">
            <div className="run-dataset-chip-row">
              {datasets.map((dataset) => (
                <label className="run-dataset-choice" key={dataset.name}>
                  <input
                    checked={selectedDatasets.includes(dataset.name)}
                    name="runDataset"
                    onChange={() => toggleDataset(dataset.name)}
                    type="checkbox"
                    value={dataset.name}
                  />
                  <span>{dataset.name}</span>
                </label>
              ))}
              {!loadingOptions && datasets.length === 0 ? <p>{t("noConfiguredDatasets")}</p> : null}
            </div>
            <span className="run-selected-count">{selectedDatasets.length} selected</span>
          </div>
        </section>

        <section className="run-flow-step retrieval-step">
          <div className="run-step-label">
            <span className="run-step-number">2</span>
            <div>
              <h2>{t("retrievalSettings")}</h2>
              <p>{t("selectRetrievalSettings")}</p>
            </div>
          </div>
          <div className="run-step-content run-settings-grid run-settings-bounded">
            <div className="run-setting-cell">
              <label>
                <span>{t("recallChunks")}</span>
                <select
                  aria-label="Recall chunks"
                  onChange={(event) => setSelectedPageSize(event.target.value)}
                  value={selectedPageSize}
                >
                  {runOptions.page_sizes.map((pageSize) => (
                    <option key={pageSize} value={String(pageSize)}>
                      {pageSize}
                    </option>
                  ))}
                  <option value={CUSTOM_OPTION}>Custom...</option>
                </select>
              </label>
              {selectedPageSize === CUSTOM_OPTION ? (
                <label>
                  <span>{t("customRecallChunks")}</span>
                  <input
                    aria-label="Custom recall chunks"
                    min="1"
                    onChange={(event) => setCustomPageSize(event.target.value)}
                    step="1"
                    type="number"
                    value={customPageSize}
                  />
                </label>
              ) : null}
            </div>
            <div className="run-setting-cell">
              <label>
                <span>{t("similarityThreshold")}</span>
                <select
                  aria-label="Similarity threshold"
                  onChange={(event) => setSelectedSimilarity(event.target.value)}
                  value={selectedSimilarity}
                >
                  {runOptions.similarity_thresholds.map((similarity) => (
                    <option key={similarity} value={String(similarity)}>
                      {similarity}
                    </option>
                  ))}
                  <option value={CUSTOM_OPTION}>Custom...</option>
                </select>
              </label>
              {selectedSimilarity === CUSTOM_OPTION ? (
                <label>
                  <span>{t("customSimilarityThreshold")}</span>
                  <input
                    aria-label="Custom similarity threshold"
                    min="0"
                    onChange={(event) => setCustomSimilarity(event.target.value)}
                    step="0.01"
                    type="number"
                    value={customSimilarity}
                  />
                </label>
              ) : null}
            </div>
          </div>
        </section>

        <section className="run-flow-step run-submit-step">
          <div className="run-step-label">
            <span className="run-step-number">3</span>
            <div>
              <h2>Run</h2>
              <p>{t("runEvaluation")}</p>
            </div>
          </div>
          <div className="run-step-content run-submit-content run-submit-balanced">
            <div className="run-summary-card">
              <span>
                <strong>Datasets:</strong> {selectedDatasetSummary}
              </span>
              <span>
                <strong>Recall:</strong> {selectedPageSizeSummary}
              </span>
              <span>
                <strong>Similarity:</strong> {selectedSimilaritySummary}
              </span>
            </div>
            <div className="run-action-buttons">
              <button disabled={starting || loadingOptions} onClick={() => start("quick")} type="button">
                {t("quickEvaluation")}
              </button>
              <button disabled={starting || loadingOptions} onClick={() => start("standard")} type="button">
                {t("standardEvaluation")}
              </button>
            </div>
          </div>
        </section>
      </div>
    </section>
  );
}
