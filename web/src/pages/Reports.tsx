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
        <div className="data-table">
          {reports.map((report) => (
            <button className="data-row button-row" key={report.url} onClick={() => setSelectedReport(report)} type="button">
              <strong>{report.name}</strong>
              <span>{report.modified}</span>
            </button>
          ))}
        </div>
        {selectedReport ? <iframe className="report-frame" src={selectedReport.url} title={selectedReport.name} /> : <p>No reports found.</p>}
      </div>
    </section>
  );
}
