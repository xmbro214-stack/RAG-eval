import type { PipelineJob } from "../api/types";
import type { Translate } from "../i18n";
import { StatusBadge } from "./StatusBadge";

type JobRailProps = {
  job: PipelineJob | null;
  t: Translate;
};

export function JobRail({ job, t }: JobRailProps) {
  const status = job?.status || t("idle");
  const progress = job?.progress?.percent ?? 0;
  const completedTasks = job?.progress?.completed_tasks;
  const totalTasks = job?.progress?.total_tasks;
  const currentTask = job?.progress?.current_task;
  const progressText =
    typeof completedTasks === "number" && typeof totalTasks === "number" ? `${completedTasks}/${totalTasks}` : null;
  const latestLog = job?.log_tail?.[job.log_tail.length - 1] || t("noBackgroundJob");

  return (
    <aside className="job-rail" aria-label={t("jobStatus")}>
      <div className="job-rail-header">
        <span>{t("job")}</span>
        <StatusBadge status={status} />
      </div>
      <div className="job-progress-track">
        <div className="job-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      {progressText || currentTask ? (
        <p className="job-log-line">
          {progressText ? <strong>{progressText}</strong> : null}
          {progressText && currentTask ? " - " : null}
          {currentTask || null}
        </p>
      ) : null}
      <p className="job-log-line">{latestLog}</p>
    </aside>
  );
}
