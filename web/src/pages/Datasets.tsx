import { useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { DatasetItem } from "../api/types";

export function Datasets() {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    apiClient
      .listDatasets()
      .then((payload) => setDatasets(payload.datasets))
      .catch((exc: Error) => setError(exc.message));
  }, []);

  return (
    <section className="section-panel">
      <h1>Datasets</h1>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="toolbar-row">
        <input aria-label="Dataset name" defaultValue="" />
        <input aria-label="Dataset file" type="file" accept=".csv,.pdf,text/csv,application/pdf" />
        <button type="button">Upload</button>
      </div>
      <div className="data-table">
        {datasets.map((dataset) => (
          <div className="data-row" key={dataset.path || dataset.name}>
            <strong>{dataset.name}</strong>
            <span>{dataset.path}</span>
            <span>{dataset.rows ?? "-"}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
