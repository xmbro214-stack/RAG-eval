import { useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { ReportItem } from "../api/types";
import type { Translate } from "../i18n";

type ReportsProps = {
  t: Translate;
};

export function Reports({ t }: ReportsProps) {
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

  function handleReportSelection(url: string) {
    setSelectedReport(reports.find((report) => report.url === url) ?? null);
  }

  return (
    <section className="section-panel reports-page">
      {error ? <p className="error-text">{error}</p> : null}
      <div className="reports-flow-panel surface-panel">
        <section className="reports-flow-step select-report-step">
          <div className="reports-step-label">
            <span className="reports-step-number">1</span>
            <div>
              <h2>{t("selectReport")}</h2>
              <p>{t("navReportsSubtitle")}</p>
            </div>
          </div>
          <div className="reports-step-content">
            {reports.length > 0 ? (
              <label className="report-select-field">
                <span>{t("selectReport")}</span>
                <select
                  aria-label="Select report"
                  onChange={(event) => handleReportSelection(event.target.value)}
                  value={selectedReport?.url ?? ""}
                >
                  {reports.map((report) => (
                    <option key={report.url} value={report.url}>
                      {report.name} - {report.modified}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <p className="muted-text">{t("noReportsFound")}</p>
            )}
          </div>
        </section>

        <section className="reports-flow-step report-details-step">
          <div className="reports-step-label">
            <span className="reports-step-number">2</span>
            <div>
              <h2>{t("reportDetails")}</h2>
              <p>{t("report")}</p>
            </div>
          </div>
          <div className="reports-step-content">
            {selectedReport ? (
              <dl className="report-flow-detail-list">
                <div>
                  <dt>{t("report")}</dt>
                  <dd>{selectedReport.name}</dd>
                </div>
                <div>
                  <dt>{t("reportPath")}</dt>
                  <dd>{selectedReport.path}</dd>
                </div>
                <div>
                  <dt>{t("reportModified")}</dt>
                  <dd>{selectedReport.modified}</dd>
                </div>
              </dl>
            ) : (
              <p className="muted-text">{t("noReportsFound")}</p>
            )}
          </div>
        </section>

        <section className="reports-flow-step open-report-step">
          <div className="reports-step-label">
            <span className="reports-step-number">3</span>
            <div>
              <h2>{t("openReport")}</h2>
              <p>{t("reports")}</p>
            </div>
          </div>
          <div className="reports-step-content report-open-step-content">
            {selectedReport ? (
              <>
                <div className="report-open-summary">
                  <strong>{selectedReport.name}</strong>
                  <span>{selectedReport.modified}</span>
                </div>
                <a className="report-open-link" href={selectedReport.url} rel="noreferrer" target="_blank">
                  {t("openReport")}
                </a>
              </>
            ) : (
              <p className="muted-text">{t("noReportsFound")}</p>
            )}
          </div>
        </section>
      </div>
    </section>
  );
}
