import { useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { EvalRunItem } from "../api/types";

export function EvaluationRuns() {
  const [runs, setRuns] = useState<EvalRunItem[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    apiClient.listEvalRuns().then(setRuns).catch((exc: Error) => setError(exc.message));
  }, []);

  return (
    <section className="section-panel">
      <h1>Evaluation Runs</h1>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="data-table">
        {runs.map((run, index) => (
          <div className="data-row" key={`${run.run_name || "run"}-${index}`}>
            <strong>{String(run.run_name || "-")}</strong>
            <span>{String(run.dataset || "-")}</span>
            {run.report_url ? <a href={String(run.report_url)}>Report</a> : <span>No report</span>}
          </div>
        ))}
      </div>
    </section>
  );
}
