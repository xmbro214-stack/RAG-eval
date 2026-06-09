import { useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { ReportItem } from "../api/types";

export function Reports() {
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [selectedReport, setSelectedReport] = useState<ReportItem | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiClient
      .listReports()
      .then((items) => {
        setReports(items);
        setSelectedReport(items[0] ?? null);
      })
      .catch((exc: Error) => setError(exc.message));
  }, []);

  return (
    <section className="section-panel">
      <h1>Reports</h1>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="split-panel">
        <div className="reports-list">
          {reports.map((report) => (
            <button
              className={selectedReport?.url === report.url ? "report-row active" : "report-row"}
              key={report.url}
              onClick={() => setSelectedReport(report)}
              type="button"
            >
              <span className="report-row-main">
                <strong className="report-row-name">{report.name}</strong>
                <span className="report-row-date">{report.modified}</span>
              </span>
              <span className="report-row-path" title={report.url}>
                {report.url}
              </span>
            </button>
          ))}
        </div>
        {selectedReport ? <iframe className="report-frame" src={selectedReport.url} title={selectedReport.name} /> : <p>No reports found.</p>}
      </div>
    </section>
  );
}
