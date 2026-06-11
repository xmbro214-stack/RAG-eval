import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

function datasetOption(select: HTMLElement, prefix: string) {
  const option = Array.from(select.querySelectorAll("option")).find((item) =>
    (item.textContent || "").startsWith(prefix)
  );

  if (!option) {
    throw new Error(`Missing dataset option starting with ${prefix}`);
  }

  return option;
}

function makeJsonResponse(payload: object) {
  return {
    ok: true,
    json: async () => payload
  };
}

function createDeferredResponse(payload: object) {
  let resolveResponse: (value: ReturnType<typeof makeJsonResponse>) => void = () => {};
  const promise = new Promise<ReturnType<typeof makeJsonResponse>>((resolve) => {
    resolveResponse = resolve;
  });

  return {
    promise,
    resolve: () => resolveResponse(makeJsonResponse(payload))
  };
}

beforeEach(() => {
  let supportQaUploaded = false;

  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);

      if (path === "/api/reports") {
        return makeJsonResponse({
          ok: true,
          reports: [
            {
              name: "smoke.html",
              path: "reports/smoke.html",
              url: "/reports/smoke.html",
              modified: "2026-06-09"
            }
          ]
        });
      }

      if (path === "/api/eval-runs") {
        return makeJsonResponse({
          ok: true,
          runs: [
            {
              run_name: "smoke",
              dataset: "qa",
              report_url: "/reports/smoke.html",
              report_modified: "2026-06-09 12:34"
            }
          ]
        });
      }

      if (path === "/api/datasets") {
        return makeJsonResponse({
          ok: true,
          datasets: [
            { name: "custom", path: "data/custom.csv", rows: 1, source: "config", runnable: true },
            { name: "trd", path: "data/trd.csv", rows: 1, source: "config", runnable: true },
            { name: "qa_golden", path: "data/qa_golden.csv", rows: 9, source: "config", runnable: false },
            {
              name: "functional_smoke_20260604_100733",
              path: "data/uploaded_datasets/functional_smoke_20260604_100733.csv",
              rows: 2,
              source: "uploaded",
              runnable: false
            },
            ...(supportQaUploaded ? [{ name: "support_qa", path: "data/support_qa.csv", rows: 2 }] : [])
          ],
          run_options: { page_sizes: [5, 10], similarity_thresholds: [0.1, 0.2] }
        });
      }

      if (path === "/api/datasets/preview?path=data%2Fcustom.csv") {
        return makeJsonResponse({
          ok: true,
          rows: 1,
          preview_rows: [{ query_id: "q1", query: "What is A?", expected_answer: "Answer A" }]
        });
      }

      if (path === "/api/datasets/preview?path=data%2Fsupport_qa.csv") {
        return makeJsonResponse({
          ok: true,
          rows: 2,
          preview_rows: [{ query_id: "q2", query: "What is B?", expected_answer: "Answer B" }]
        });
      }

      if (path === "/api/datasets/preview?path=data%2Fqa_golden.csv") {
        return makeJsonResponse({
          ok: true,
          rows: 9,
          preview_rows: [{ query_id: "q9", query: "What is CU64 M1?", expected_answer: "CU64 M1 answer" }]
        });
      }

      if (path === "/api/datasets/upload" && init?.method === "POST") {
        supportQaUploaded = true;

        return makeJsonResponse({
          ok: true,
          dataset_name: "support_qa",
          path: "data/support_qa.csv",
          rows: 2
        });
      }

      if (path === "/api/datasets/manual-qa" && init?.method === "POST") {
        return makeJsonResponse({ ok: true, appended_query_id: "query_2" });
      }

      if (path === "/api/datasets/generate-answer" && init?.method === "POST") {
        return makeJsonResponse({
          ok: true,
          question: "What is CU64 M1?",
          expected_answer: "CU64 M1 is a generated standard answer.",
          passages: [
            {
              id: "chunk-1",
              text: "CU64 M1 is described in the retrieved passage.",
              source: "manual.pdf",
              score: 0.88
            }
          ]
        });
      }

      if (path === "/api/datasets/regenerate-answer" && init?.method === "POST") {
        return makeJsonResponse({
          ok: true,
          expected_answer: "CU64 M1 is an improved generated standard answer."
        });
      }

      if (path === "/api/pipeline/run" && init?.method === "POST") {
        return makeJsonResponse({ ok: true, run_id: "run-1", status: "starting" });
      }
      if (path === "/api/pipeline/status?run_id=run-1") {
        return makeJsonResponse({
          ok: true,
          job: {
            run_id: "run-1",
            status: "running",
            progress: {
              total_tasks: 2,
              completed_tasks: 1,
              current_task: "custom ps10 sim0.2",
              percent: 50
            },
            log_tail: ["Task 1 started", "custom ps10 sim0.2"]
          }
        });
      }

      return makeJsonResponse({ ok: true });
    })
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("App", () => {
  it("marks the shell with the Apple-inspired blue visual theme", async () => {
    const { container } = render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
    expect(container.querySelector(".app-shell")).toHaveClass("apple-blue-theme");
  });
  it("places job status in the global top bar with the AT&S mark", async () => {
    render(<App />);

    const nav = screen.getByLabelText("Console sections");
    const jobStatus = screen.getByLabelText("Job status");
    const topbar = jobStatus.parentElement;
    const brandLogo = screen.getByText("AT&S");

    expect(brandLogo).toHaveClass("brand-logo");
    expect(nav).toContainElement(brandLogo);
    expect(topbar).toHaveClass("global-topbar");
    expect(topbar).not.toContainElement(brandLogo);
    expect(topbar?.querySelector(".page-title")).toHaveTextContent("Datasets");
    expect(topbar?.querySelector(".page-subtitle")).toHaveTextContent("Manage QA datasets");
    expect(await screen.findByText("custom")).toBeInTheDocument();
  });

  it("renders Datasets as a compact D2 control bar with the QA workspace as the main area", async () => {
    const { container } = render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
    const datasetsLayout = container.querySelector(".datasets-layout");
    const controlBar = container.querySelector(".dataset-control-bar");
    const workspacePanel = container.querySelector(".dataset-workspace-panel");
    const manualQaCard = container.querySelector(".manual-qa-card");
    const disclosureStack = container.querySelector(".dataset-disclosure-stack");

    expect(container.querySelector(".datasets-page")).not.toBeNull();
    expect(datasetsLayout).toHaveClass("datasets-d2-layout");
    expect(controlBar).toHaveClass("surface-panel", "dataset-console-card");
    expect(screen.getByLabelText("Use dataset")).toBeInTheDocument();
    expect(screen.queryByLabelText("Save target dataset")).not.toBeInTheDocument();
    expect(screen.getByText("Upload dataset")).toBeInTheDocument();
    expect(container.querySelector(".dataset-list")).toBeNull();
    expect(workspacePanel).not.toBeNull();
    expect(manualQaCard).toHaveClass("surface-panel");
    expect(disclosureStack).not.toBeNull();
    expect(disclosureStack).toContainElement(container.querySelector(".dataset-details-panel"));
    expect(disclosureStack).toContainElement(container.querySelector(".retrieved-passages-panel"));
    expect(manualQaCard).not.toContainElement(container.querySelector(".dataset-details-panel"));
    expect(controlBar?.compareDocumentPosition(workspacePanel as Element)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it("hides uploaded dataset artifacts from the dataset selectors", async () => {
    render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /functional_smoke_20260604_100733/i })).not.toBeInTheDocument();
    expect(screen.queryByText("data/uploaded_datasets/functional_smoke_20260604_100733.csv")).not.toBeInTheDocument();
  });

  it("uses unified surface panels for the simplified console layout", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
    expect(container.querySelector(".dataset-control-bar")).toHaveClass("surface-panel");
    expect(container.querySelector(".primary-qa-panel")).toHaveClass("surface-panel");

    await user.click(screen.getByRole("button", { name: "Run" }));
    expect(await screen.findByLabelText("custom")).toBeInTheDocument();
    expect(container.querySelector(".run-flow-panel")).toHaveClass("surface-panel");
    expect(container.querySelectorAll(".run-flow-step")).toHaveLength(3);
    expect(container.querySelector(".run-control-grid")).toBeNull();
    expect(container.querySelector(".run-actions-bar")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Reports" }));
    expect((await screen.findAllByText("smoke.html")).length).toBeGreaterThan(0);
    expect(container.querySelector(".reports-flow-panel")).toHaveClass("surface-panel");
    expect(container.querySelector(".report-details-step")).not.toBeNull();
  });

  it("emphasizes Question and Expected answer as the primary dataset workspace", async () => {
    const { container } = render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
    expect(container.querySelector(".manual-qa-primary-grid")).toHaveClass("manual-qa-symmetric-grid");
    expect(screen.getByLabelText("Question")).toHaveClass("qa-main-textarea");
    expect(screen.getByLabelText("Expected answer")).toHaveClass("qa-main-textarea");
    expect(container.querySelector(".dataset-details-panel")).toHaveClass("auxiliary-panel");
    expect(container.querySelector(".retrieved-passages-panel")).toHaveClass("auxiliary-panel");
  });


  it("pairs each Manual Q&A textarea with its own right-aligned action", async () => {
    const { container } = render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
    const questionColumn = container.querySelector(".manual-qa-column.question-column");
    const answerColumn = container.querySelector(".manual-qa-column.answer-column");

    expect(questionColumn).toContainElement(screen.getByLabelText("Question"));
    expect(questionColumn).toContainElement(screen.getByRole("button", { name: "Generate Answer" }));
    expect(answerColumn).toContainElement(screen.getByLabelText("Expected answer"));
    expect(answerColumn).toContainElement(screen.getByRole("button", { name: "Save Q&A" }));
    expect(container.querySelector(".manual-qa-column-divider")).not.toBeNull();
    expect(container.querySelector(".manual-qa-button-row")).toBeNull();
  });
  it("loads datasets on the default page", async () => {
    render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
  });

  it("renders language switcher as a top bar select in German, English, Chinese, Malay order", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    const languageSelect = screen.getByLabelText("Language");
    expect(languageSelect.closest(".global-topbar")).not.toBeNull();
    expect(container.querySelector(".sidebar-nav .language-switcher")).toBeNull();
    /*
    expect(
      screen
        .getAllByRole("option")
        .filter((option) => /^(Deutsch|English|娑擃厽鏋億Bahasa Melayu)$/.test(option.textContent || ""))
        .map((option) => option.textContent)
    ).toEqual(["Deutsch", "English", "娑擃厽鏋?, "Bahasa Melayu"]);

    await user.selectOptions(languageSelect, "zh");

    expect(screen.getByRole("button", { name: "閺佺増宓侀梿? })).toBeInTheDocument();
  });

    */
    expect(Array.from(languageSelect.querySelectorAll("option")).map((option) => option.value)).toEqual([
      "de",
      "en",
      "zh",
      "ms"
    ]);
    await user.selectOptions(languageSelect, "zh");
    expect(languageSelect).toHaveValue("zh");
  });

  it("keeps Save Q&A under the expected answer field without Regenerate Answer", async () => {
    render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
    const generateButton = screen.getByRole("button", { name: "Generate Answer" });
    const saveButton = screen.getByRole("button", { name: "Save Q&A" });
    const generateRow = generateButton.closest(".qa-field-action-row");
    const saveRow = saveButton.closest(".qa-field-action-row");

    expect(screen.queryByRole("button", { name: "Regenerate Answer" })).not.toBeInTheDocument();
    expect(generateRow).toContainElement(generateButton);
    expect(generateRow).not.toContainElement(saveButton);
    expect(saveRow).toContainElement(saveButton);
    expect(saveRow).not.toContainElement(generateButton);
    expect(saveButton).toHaveClass("manual-qa-save-action");
    expect(saveRow).toHaveClass("qa-field-action-row");
  });

  it("loads dataset preview when selecting a dataset", async () => {
    const user = userEvent.setup();
    render(<App />);

    const datasetSelect = screen.getByLabelText("Use dataset");
    await waitFor(() => expect(datasetSelect).not.toBeDisabled());
    await user.selectOptions(datasetSelect, datasetOption(datasetSelect, "custom - data/custom.csv"));

    expect(await screen.findByText("What is A?")).toBeInTheDocument();
    expect(screen.getByText("1 rows")).toBeInTheDocument();
  });

  it("renders dataset choices as compact select options instead of rows", async () => {
    render(<App />);

    const datasetSelect = screen.getByLabelText("Use dataset");
    await waitFor(() => expect(datasetSelect).not.toBeDisabled());
    expect(datasetOption(datasetSelect, "custom - data/custom.csv")).toBeInTheDocument();
    expect(datasetSelect).toHaveClass("dataset-select");
    expect(screen.queryByLabelText("Save target dataset")).not.toBeInTheDocument();
  });

  it("keeps the selected dataset preview when preview responses arrive out of order", async () => {
    const user = userEvent.setup();
    const customPreview = createDeferredResponse({
      ok: true,
      rows: 1,
      preview_rows: [{ query_id: "q1", query: "What is A?", expected_answer: "Answer A" }]
    });
    const supportPreview = createDeferredResponse({
      ok: true,
      rows: 2,
      preview_rows: [{ query_id: "q2", query: "What is B?", expected_answer: "Answer B" }]
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = String(input);

        if (path === "/api/datasets") {
          return makeJsonResponse({
            ok: true,
            datasets: [
              { name: "custom", path: "data/custom.csv", rows: 1 },
              { name: "support_qa", path: "data/support_qa.csv", rows: 2 }
            ],
            run_options: { page_sizes: [5], similarity_thresholds: [0.1] }
          });
        }

        if (path === "/api/datasets/preview?path=data%2Fcustom.csv") {
          return customPreview.promise;
        }

        if (path === "/api/datasets/preview?path=data%2Fsupport_qa.csv") {
          return supportPreview.promise;
        }

        return makeJsonResponse({ ok: true });
      })
    );

    render(<App />);

    const datasetSelect = await screen.findByLabelText("Use dataset");
    await waitFor(() => expect(datasetSelect).not.toBeDisabled());
    await user.selectOptions(datasetSelect, datasetOption(datasetSelect, "custom - data/custom.csv"));
    await user.selectOptions(
      datasetSelect,
      datasetOption(datasetSelect, "support_qa - data/support_qa.csv")
    );

    await act(async () => {
      supportPreview.resolve();
      await supportPreview.promise;
    });
    expect(await screen.findByText("What is B?")).toBeInTheDocument();

    await act(async () => {
      customPreview.resolve();
      await customPreview.promise;
    });
    expect(screen.getByRole("heading", { name: "support_qa" })).toBeInTheDocument();
    expect(screen.getByText("What is B?")).toBeInTheDocument();
    expect(screen.queryByText("What is A?")).not.toBeInTheDocument();
  });

  it("treats datasets with the same path and different ids as distinct options", async () => {
    const user = userEvent.setup();

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = String(input);

        if (path === "/api/datasets") {
          return makeJsonResponse({
            ok: true,
            datasets: [
              {
                name: "support_primary",
                path: "data/shared.csv",
                rows: 3,
                source: "config",
                dataset_id: "support_primary"
              },
              {
                name: "support_regression",
                path: "data/shared.csv",
                rows: 3,
                source: "config",
                dataset_id: "support_regression"
              }
            ],
            run_options: { page_sizes: [5], similarity_thresholds: [0.1] }
          });
        }

        if (path === "/api/datasets/preview?path=data%2Fshared.csv") {
          return makeJsonResponse({
            ok: true,
            rows: 3,
            preview_rows: [{ query_id: "q1", query: "Shared path question?", expected_answer: "Shared answer" }]
          });
        }

        return makeJsonResponse({ ok: true });
      })
    );

    render(<App />);

    const datasetSelect = screen.getByLabelText("Use dataset");
    await waitFor(() => expect(datasetSelect).not.toBeDisabled());
    const primary = datasetOption(datasetSelect, "support_primary - data/shared.csv");
    const regression = datasetOption(datasetSelect, "support_regression - data/shared.csv");
    await user.selectOptions(datasetSelect, regression);

    expect(datasetSelect).toHaveDisplayValue("support_regression - data/shared.csv (3 rows)");
    expect(primary).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "support_regression" })).toBeInTheDocument();
  });

  it("uploads a dataset and refreshes inventory", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.upload(
      screen.getByLabelText("Dataset file"),
      new File(["query_id,query,expected_answer\nq1,What is A?,Answer A\n"], "qa.csv", { type: "text/csv" })
    );

    expect(await screen.findByText("Uploaded support_qa")).toBeInTheDocument();
    const datasetSelect = screen.getByLabelText("Use dataset");
    expect(datasetOption(datasetSelect, "support_qa - data/support_qa.csv")).toBeInTheDocument();
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock.mock.calls.filter(([input]) => String(input) === "/api/datasets")).toHaveLength(2);
  });

  it("renders upload as a single compact file action", async () => {
    render(<App />);

    expect(await screen.findByText("Upload dataset")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Dataset name (optional)")).not.toBeInTheDocument();
    expect(screen.queryByText("No file selected")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Dataset file")).toHaveClass("hidden-file-input");
  });

  it("saves manual Q&A to the selected dataset", async () => {
    const user = userEvent.setup();
    render(<App />);

    const datasetSelect = screen.getByLabelText("Use dataset");
    await waitFor(() => expect(datasetSelect).not.toBeDisabled());
    await user.selectOptions(datasetSelect, datasetOption(datasetSelect, "custom - data/custom.csv"));
    await user.type(screen.getByLabelText("Question"), "What is B?");
    await user.type(screen.getByLabelText("Expected answer"), "Answer B");
    await user.click(screen.getByRole("button", { name: "Save Q&A" }));

    expect(await screen.findByText("Saved query_2")).toBeInTheDocument();
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock).toHaveBeenCalledWith("/api/datasets/manual-qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_dataset: "data/custom.csv",
        question: "What is B?",
        expected_answer: "Answer B"
      })
    });
  });

  it("saves manual Q&A to the currently selected dataset", async () => {
    const user = userEvent.setup();
    render(<App />);

    const datasetSelect = screen.getByLabelText("Use dataset");
    await waitFor(() => expect(datasetSelect).not.toBeDisabled());
    await user.selectOptions(datasetSelect, datasetOption(datasetSelect, "qa_golden - data/qa_golden.csv"));
    await user.type(screen.getByLabelText("Question"), "What is CU64 M1?");
    await user.type(screen.getByLabelText("Expected answer"), "CU64 M1 is a ceramic unit package.");
    const saveButton = document.querySelector(".manual-qa-save-action") as HTMLElement;
    await user.click(saveButton);

    const fetchMock = vi.mocked(fetch);
    expect(fetchMock).toHaveBeenCalledWith("/api/datasets/manual-qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_dataset: "data/qa_golden.csv",
        question: "What is CU64 M1?",
        expected_answer: "CU64 M1 is a ceramic unit package."
      })
    });
  });


  it("generates and saves a grounded Q&A answer from the language select", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText("Language"), "zh");
    const datasetSelect = screen.getByLabelText("Use dataset");
    await waitFor(() => expect(datasetSelect).not.toBeDisabled());
    await user.selectOptions(datasetSelect, datasetOption(datasetSelect, "custom - data/custom.csv"));
    await user.type(screen.getByLabelText("Question"), "What is CU64 M1?");
        const generateButton = document.querySelector(".manual-qa-generate-action") as HTMLElement;
    await user.click(generateButton);

    expect(await screen.findByDisplayValue("CU64 M1 is a generated standard answer.")).toBeInTheDocument();
    expect(screen.getByText("CU64 M1 is described in the retrieved passage.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "重新生成回答" })).not.toBeInTheDocument();

        const saveButton = document.querySelector(".manual-qa-save-action") as HTMLElement;
    await user.click(saveButton);

    const fetchMock = vi.mocked(fetch);
    expect(fetchMock).toHaveBeenCalledWith("/api/datasets/generate-answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "What is CU64 M1?", language: "zh" })
    });
    expect(fetchMock).not.toHaveBeenCalledWith("/api/datasets/regenerate-answer", expect.anything());
    expect(fetchMock).toHaveBeenCalledWith("/api/datasets/manual-qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_dataset: "data/custom.csv",
        question: "What is CU64 M1?",
        expected_answer: "CU64 M1 is a generated standard answer."
      })
    });
  });

  it("renders Reports as a vertical flow instead of a split panel", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.click(screen.getByRole("button", { name: "Reports" }));

    expect(await screen.findByLabelText("Select report")).toHaveDisplayValue("smoke.html - 2026-06-09");
    expect(container.querySelector(".reports-flow-panel")).toHaveClass("surface-panel");
    expect(container.querySelectorAll(".reports-flow-step")).toHaveLength(3);
    expect(container.querySelector(".split-panel")).toBeNull();
    expect(container.querySelector(".reports-list")).toBeNull();
    expect(container.querySelector(".report-detail-panel")).toBeNull();
    expect(container.querySelector(".reports-flow-step.select-report-step")?.compareDocumentPosition(
      container.querySelector(".reports-flow-step.report-details-step") as Element
    )).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(container.querySelector(".reports-flow-step.report-details-step")?.compareDocumentPosition(
      container.querySelector(".reports-flow-step.open-report-step") as Element
    )).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it("opens reports from the final Reports flow step without embedding the console", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.click(screen.getByRole("button", { name: "Reports" }));

    expect(await screen.findByText("Report details")).toBeInTheDocument();
    expect(screen.getAllByText("smoke.html").length).toBeGreaterThan(0);
    expect(screen.getByText("reports/smoke.html")).toBeInTheDocument();
    expect(screen.getAllByText("2026-06-09").length).toBeGreaterThan(0);
    expect(container.querySelector(".report-frame")).toBeNull();

    const openReport = screen.getByRole("link", { name: "Open report" });
    expect(openReport).toHaveAttribute("href", "/reports/smoke.html");
    expect(openReport).toHaveAttribute("target", "_blank");
    expect(openReport.closest(".open-report-step")).not.toBeNull();
  });

  
  it("loads evaluation runs when switching to Records", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Records" }));

    expect(await screen.findByText("smoke")).toBeInTheDocument();
    expect(screen.getByText("2026-06-09 12:34")).toBeInTheDocument();
  });

  it("shows run controls", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.click(screen.getByRole("button", { name: "Run" }));

    expect(await screen.findByLabelText("custom")).toBeChecked();
    expect(container.querySelector(".run-page")).not.toBeNull();
    expect(container.querySelector(".run-flow-panel")).not.toBeNull();
    expect(container.querySelector(".run-flow-panel")).toHaveClass("run-flow-wide-panel");
    expect(container.querySelector(".run-settings-grid")).toHaveClass("run-settings-bounded");
    expect(container.querySelector(".run-submit-content")).toHaveClass("run-submit-balanced");
    expect(container.querySelectorAll(".run-flow-step")).toHaveLength(3);
    expect(container.querySelector(".run-flow-step.datasets-step")?.compareDocumentPosition(
      container.querySelector(".run-flow-step.retrieval-step") as Element
    )).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(container.querySelector(".run-flow-step.retrieval-step")?.compareDocumentPosition(
      container.querySelector(".run-flow-step.run-submit-step") as Element
    )).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(container.querySelector(".run-control-grid")).toBeNull();
    expect(screen.getByLabelText("trd")).toBeChecked();
    expect(screen.getByLabelText("Recall chunks")).toHaveDisplayValue("5");
    expect(screen.getByLabelText("Similarity threshold")).toHaveDisplayValue("0.1");
    expect(screen.getByText("2 selected")).toBeInTheDocument();
    expect(screen.getByText("Datasets:")).toBeInTheDocument();
    expect(screen.getByText("custom, trd")).toBeInTheDocument();
    expect(screen.getByText("Recall:")).toBeInTheDocument();
    expect(screen.getAllByText("5").length).toBeGreaterThan(0);
    expect(screen.getByText("Similarity:")).toBeInTheDocument();
    expect(screen.getAllByText("0.1").length).toBeGreaterThan(0);

    await user.click(screen.getByLabelText("trd"));
    await user.selectOptions(screen.getByLabelText("Recall chunks"), "10");
    await user.selectOptions(screen.getByLabelText("Similarity threshold"), "0.2");
    expect(screen.getByText("1 selected")).toBeInTheDocument();
    expect(screen.getAllByText("custom").length).toBeGreaterThan(0);
    expect(screen.getAllByText("10").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.2").length).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: "Standard evaluation" }));

    expect(await screen.findByText("Started run-1")).toBeInTheDocument();
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock).toHaveBeenCalledWith("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "standard",
        stage: "all",
        datasets: ["custom"],
        page_sizes: [10],
        similarity_thresholds: [0.2]
      })
    });
  });

  it("runs evaluation with custom recall chunks and similarity threshold", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Run" }));

    await screen.findByLabelText("custom");
    await user.selectOptions(screen.getByLabelText("Recall chunks"), "custom");
    await user.clear(screen.getByLabelText("Custom recall chunks"));
    await user.type(screen.getByLabelText("Custom recall chunks"), "7");
    await user.selectOptions(screen.getByLabelText("Similarity threshold"), "custom");
    await user.clear(screen.getByLabelText("Custom similarity threshold"));
    await user.type(screen.getByLabelText("Custom similarity threshold"), "0.15");
    await user.click(screen.getByRole("button", { name: "Standard evaluation" }));

    expect(await screen.findByText("Started run-1")).toBeInTheDocument();
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock).toHaveBeenCalledWith("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "standard",
        stage: "all",
        datasets: ["custom", "trd"],
        page_sizes: [7],
        similarity_thresholds: [0.15]
      })
    });
  });

  it("polls the active pipeline job and shows progress details", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByLabelText("custom");
    await user.click(screen.getByRole("button", { name: "Standard evaluation" }));

    expect(await screen.findByText("1/2")).toBeInTheDocument();
    expect(screen.getByText("custom ps10 sim0.2")).toBeInTheDocument();
  });
  it("renders Records, Reports, and Overview with simplified page shells", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.click(screen.getByRole("button", { name: "Records" }));
    expect(await screen.findByText("smoke")).toBeInTheDocument();
    expect(container.querySelector(".records-page")).not.toBeNull();
    expect(container.querySelector(".records-list-panel")).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "Reports" }));
    expect((await screen.findAllByText("smoke.html")).length).toBeGreaterThan(0);
    expect(container.querySelector(".reports-page")).not.toBeNull();
    expect(container.querySelector(".reports-flow-panel")).not.toBeNull();
    expect(container.querySelector(".report-details-step")).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "Overview" }));
    expect(container.querySelector(".overview-page")).not.toBeNull();
    expect(container.querySelector(".overview-status-card")).not.toBeNull();
  });
});





