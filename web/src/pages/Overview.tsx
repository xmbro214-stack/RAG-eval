import type { PipelineJob } from "../api/types";

type OverviewProps = {
  job: PipelineJob | null;
};

export function Overview({ job }: OverviewProps) {
  return (
    <section className="section-panel">
      <h1>Overview</h1>
      <div className="summary-grid">
        <div>
          <span>Active job</span>
          <strong>{job?.status || "Idle"}</strong>
        </div>
      </div>
    </section>
  );
}
