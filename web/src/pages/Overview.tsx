import type { PipelineJob } from "../api/types";
import type { Translate } from "../i18n";

type OverviewProps = {
  job: PipelineJob | null;
  t: Translate;
};

export function Overview({ job, t }: OverviewProps) {
  return (
    <section className="section-panel overview-page">
      <div className="summary-grid">
        <div className="overview-status-card surface-panel">
          <span>{t("activeJob")}</span>
          <strong>{job?.status || t("idle")}</strong>
        </div>
      </div>
    </section>
  );
}
