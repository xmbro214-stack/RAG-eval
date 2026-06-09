import { useState } from "react";

import type { PipelineJob } from "./api/types";
import { AppShell } from "./components/AppShell";
import "./styles/globals.css";

type SectionId = "datasets" | "run" | "records" | "reports" | "overview";

const titles: Record<SectionId, string> = {
  datasets: "Datasets",
  run: "Run Evaluation",
  records: "Evaluation Runs",
  reports: "Reports",
  overview: "Overview"
};

export function App() {
  const [activeSection, setActiveSection] = useState<SectionId>("datasets");
  const [job] = useState<PipelineJob | null>(null);

  return (
    <AppShell activeSection={activeSection} job={job} onSectionChange={setActiveSection}>
      <section className="section-panel">
        <h1>{titles[activeSection]}</h1>
      </section>
    </AppShell>
  );
}
