import type { ReactNode } from "react";

import type { PipelineJob } from "../api/types";
import { JobRail } from "./JobRail";

type SectionId = "datasets" | "run" | "records" | "reports" | "overview";

type AppShellProps = {
  activeSection: SectionId;
  job: PipelineJob | null;
  onSectionChange: (section: SectionId) => void;
  children: ReactNode;
};

const navItems: Array<{ id: SectionId; label: string }> = [
  { id: "datasets", label: "Datasets" },
  { id: "run", label: "Run" },
  { id: "records", label: "Records" },
  { id: "reports", label: "Reports" },
  { id: "overview", label: "Overview" }
];

export function AppShell({ activeSection, job, onSectionChange, children }: AppShellProps) {
  return (
    <div className="app-shell">
      <nav className="sidebar-nav" aria-label="Console sections">
        <div className="brand-mark">RAG Eval</div>
        {navItems.map((item) => (
          <button
            className={item.id === activeSection ? "nav-button active" : "nav-button"}
            key={item.id}
            onClick={() => onSectionChange(item.id)}
            type="button"
          >
            {item.label}
          </button>
        ))}
      </nav>
      <main className="workspace">{children}</main>
      <JobRail job={job} />
    </div>
  );
}
