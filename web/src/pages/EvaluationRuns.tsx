import { useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { EvalRunItem } from "../api/types";
import type { Translate } from "../i18n";

type EvaluationRunsProps = {
  t: Translate;
};

export function EvaluationRuns({ t }: EvaluationRunsProps) {
  const [runs, setRuns] = useState<EvalRunItem[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    apiClient.listEvalRuns().then(setRuns).catch((exc: Error) => setError(exc.message));
  }, []);

  return (
    <section className="section-panel records-page">
      {error ? <p className="error-text">{error}</p> : null}
      <div className="data-table records-list-panel surface-panel">
        {runs.map((run, index) => {
          const reportTime = run.report_url ? String(run.report_modified || "-") : "-";

          return (
            <div className="data-row" key={`${run.run_name || "run"}-${index}`}>
              <strong>{String(run.run_name || "-")}</strong>
              <span className="records-report-time">{reportTime}</span>
              {run.report_url ? <a href={String(run.report_url)}>{t("report")}</a> : <span>{t("noReport")}</span>}
            </div>
          );
        })}
      </div>
    </section>
  );
}
