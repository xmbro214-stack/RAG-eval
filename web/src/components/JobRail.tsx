import type { PipelineJob } from "../api/types";
import { StatusBadge } from "./StatusBadge";

type JobRailProps = {
  job: PipelineJob | null;
};

export function JobRail({ job }: JobRailProps) {
  const status = job?.status || "Idle";
  const progress = job?.progress?.percent ?? 0;
  const latestLog = job?.log_tail?.at(-1) || "No background job is running.";

  return (
    <aside className="job-rail" aria-label="Job status">
      <div className="job-rail-header">
        <span>Job</span>
        <StatusBadge status={status} />
      </div>
      <div className="job-progress-track">
        <div className="job-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <p className="job-log-line">{latestLog}</p>
    </aside>
  );
}
