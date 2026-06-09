import { useEffect, useState } from "react";

import type { PipelineJob } from "./api/types";
import { AppShell } from "./components/AppShell";
import { Datasets } from "./pages/Datasets";
import { EvaluationRuns } from "./pages/EvaluationRuns";
import { Overview } from "./pages/Overview";
import { Reports } from "./pages/Reports";
import { RunEvaluation } from "./pages/RunEvaluation";
import "./styles/globals.css";

type SectionId = "datasets" | "run" | "records" | "reports" | "overview";

export function App() {
  const [activeSection, setActiveSection] = useState<SectionId>("datasets");
  const [job, setJob] = useState<PipelineJob | null>(null);

  useEffect(() => {
    if (job?.run_id) {
      window.localStorage.setItem("ragEvalActiveRunId", job.run_id);
    }
  }, [job?.run_id]);

  return (
    <AppShell activeSection={activeSection} job={job} onSectionChange={setActiveSection}>
      {activeSection === "datasets" ? <Datasets /> : null}
      {activeSection === "run" ? <RunEvaluation onJobStart={setJob} /> : null}
      {activeSection === "records" ? <EvaluationRuns /> : null}
      {activeSection === "reports" ? <Reports /> : null}
      {activeSection === "overview" ? <Overview job={job} /> : null}
    </AppShell>
  );
}
