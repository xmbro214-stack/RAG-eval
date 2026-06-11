import { useEffect, useState } from "react";

import { apiClient } from "./api/client";
import type { PipelineJob } from "./api/types";
import { AppShell } from "./components/AppShell";
import { languageOptions, translate, type LanguageCode, type TranslationKey } from "./i18n";
import { Datasets } from "./pages/Datasets";
import { EvaluationRuns } from "./pages/EvaluationRuns";
import { Overview } from "./pages/Overview";
import { Reports } from "./pages/Reports";
import { RunEvaluation } from "./pages/RunEvaluation";
import "./styles/globals.css";

type SectionId = "datasets" | "run" | "records" | "reports" | "overview";
const ACTIVE_JOB_STATUSES = new Set(["starting", "running", "cancelling"]);
const JOB_POLL_INTERVAL_MS = 2000;

export function App() {
  const [activeSection, setActiveSection] = useState<SectionId>("datasets");
  const [job, setJob] = useState<PipelineJob | null>(null);
  const [language, setLanguage] = useState<LanguageCode>("en");
  const t = (key: TranslationKey) => translate(language, key);

  useEffect(() => {
    const storedRunId = window.localStorage.getItem("ragEvalActiveRunId");
    if (storedRunId) {
      setJob({ run_id: storedRunId, status: "running" });
    }
  }, []);

  useEffect(() => {
    if (job?.run_id) {
      window.localStorage.setItem("ragEvalActiveRunId", job.run_id);
    }
  }, [job?.run_id]);

  useEffect(() => {
    if (!job?.run_id) {
      return undefined;
    }

    const runId = job.run_id;
    const status = job.status;
    const hasProgress = Boolean(job.progress);
    let cancelled = false;
    let timeoutId: number | undefined;

    async function refreshJob() {
      try {
        const nextJob = await apiClient.getPipelineStatus(runId);
        if (cancelled) {
          return;
        }
        setJob(nextJob);
        if (ACTIVE_JOB_STATUSES.has(nextJob.status)) {
          timeoutId = window.setTimeout(refreshJob, JOB_POLL_INTERVAL_MS);
        }
      } catch {
        if (!cancelled && ACTIVE_JOB_STATUSES.has(status)) {
          timeoutId = window.setTimeout(refreshJob, JOB_POLL_INTERVAL_MS);
        }
      }
    }

    if (ACTIVE_JOB_STATUSES.has(status) || !hasProgress) {
      void refreshJob();
    }

    return () => {
      cancelled = true;
      if (timeoutId !== undefined) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [job?.run_id, job?.status]);

  return (
    <AppShell
      activeSection={activeSection}
      job={job}
      language={language}
      languageOptions={languageOptions}
      onLanguageChange={setLanguage}
      onSectionChange={setActiveSection}
      t={t}
    >
      {activeSection === "datasets" ? <Datasets language={language} t={t} /> : null}
      {activeSection === "run" ? <RunEvaluation onJobStart={setJob} t={t} /> : null}
      {activeSection === "records" ? <EvaluationRuns t={t} /> : null}
      {activeSection === "reports" ? <Reports t={t} /> : null}
      {activeSection === "overview" ? <Overview job={job} t={t} /> : null}
    </AppShell>
  );
}
