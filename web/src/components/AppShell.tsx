import type { ReactNode } from "react";

import type { PipelineJob } from "../api/types";
import type { LanguageCode, Translate, TranslationKey } from "../i18n";
import { JobRail } from "./JobRail";

type SectionId = "datasets" | "run" | "records" | "reports" | "overview";

type AppShellProps = {
  activeSection: SectionId;
  job: PipelineJob | null;
  language: LanguageCode;
  languageOptions: Array<{ code: LanguageCode; label: string }>;
  onLanguageChange: (language: LanguageCode) => void;
  onSectionChange: (section: SectionId) => void;
  t: Translate;
  children: ReactNode;
};

const navItems: Array<{ id: SectionId; labelKey: TranslationKey; subtitleKey: TranslationKey }> = [
  { id: "datasets", labelKey: "navDatasets", subtitleKey: "navDatasetsSubtitle" },
  { id: "run", labelKey: "navRun", subtitleKey: "navRunSubtitle" },
  { id: "records", labelKey: "navRecords", subtitleKey: "navRecordsSubtitle" },
  { id: "reports", labelKey: "navReports", subtitleKey: "navReportsSubtitle" },
  { id: "overview", labelKey: "navOverview", subtitleKey: "navOverviewSubtitle" }
];

export function AppShell({
  activeSection,
  job,
  language,
  languageOptions,
  onLanguageChange,
  onSectionChange,
  t,
  children
}: AppShellProps) {
  const activeItem = navItems.find((item) => item.id === activeSection) ?? navItems[0];

  return (
    <div className="app-shell apple-blue-theme">
      <nav className="sidebar-nav" aria-label={t("consoleSections")}>
        <div className="brand-logo" aria-label="AT&S">
          AT&S
        </div>
        {navItems.map((item) => (
          <button
            className={item.id === activeSection ? "nav-button active" : "nav-button"}
            key={item.id}
            onClick={() => onSectionChange(item.id)}
            type="button"
          >
            {t(item.labelKey)}
          </button>
        ))}
      </nav>
      <header className="global-topbar">
        <div className="page-context">
          <span className="page-eyebrow">RAG Eval Console</span>
          <h1 className="page-title">{t(activeItem.labelKey)}</h1>
          <p className="page-subtitle">{t(activeItem.subtitleKey)}</p>
        </div>
        <label className="topbar-language-field">
          <span>{t("language")}</span>
          <select
            aria-label="Language"
            onChange={(event) => onLanguageChange(event.target.value as LanguageCode)}
            value={language}
          >
            {languageOptions.map((option) => (
              <option key={option.code} value={option.code}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <JobRail job={job} t={t} />
      </header>
      <main className="workspace">{children}</main>
    </div>
  );
}
