import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

const customDatasetButtonName = /custom data\/custom\.csv 1 rows/i;

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
          runs: [{ run_name: "smoke", dataset: "qa", report_url: "/reports/smoke.html" }]
        });
      }

      if (path === "/api/datasets") {
        return makeJsonResponse({
          ok: true,
          datasets: [
            { name: "custom", path: "data/custom.csv", rows: 1 },
            ...(supportQaUploaded ? [{ name: "support_qa", path: "data/support_qa.csv", rows: 2 }] : [])
          ],
          run_options: { page_sizes: [5], similarity_thresholds: [0.1] }
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

      return makeJsonResponse({ ok: true });
    })
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("App", () => {
  it("loads datasets on the default page", async () => {
    render(<App />);

    expect(await screen.findByText("custom")).toBeInTheDocument();
  });

  it("loads dataset preview when selecting a dataset", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: customDatasetButtonName }));

    expect(await screen.findByText("What is A?")).toBeInTheDocument();
    expect(screen.getByText("1 rows")).toBeInTheDocument();
  });

  it("renders dataset rows as compact list items with truncated paths", async () => {
    render(<App />);

    const row = await screen.findByRole("button", { name: customDatasetButtonName });

    expect(row.querySelector(".dataset-row-main")).not.toBeNull();
    expect(row.querySelector(".dataset-row-path")).toHaveTextContent("data/custom.csv");
    expect(row.querySelector(".dataset-row-count")).toHaveTextContent("Rows: 1");
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

    await user.click(await screen.findByRole("button", { name: customDatasetButtonName }));
    await user.click(await screen.findByRole("button", { name: /support_qa data\/support_qa\.csv 2 rows/i }));

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

  it("treats datasets with the same path and different ids as distinct rows", async () => {
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

    const primary = await screen.findByRole("button", {
      name: /support_primary data\/shared\.csv 3 rows/i
    });
    const regression = await screen.findByRole("button", {
      name: /support_regression data\/shared\.csv 3 rows/i
    });

    await user.click(regression);

    expect(regression).toHaveClass("active");
    expect(primary).not.toHaveClass("active");
    expect(screen.getByRole("heading", { name: "support_regression" })).toBeInTheDocument();
  });

  it("uploads a dataset and refreshes inventory", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText("Dataset name"), "Support QA");
    await user.upload(
      screen.getByLabelText("Dataset file"),
      new File(["query_id,query,expected_answer\nq1,What is A?,Answer A\n"], "qa.csv", { type: "text/csv" })
    );
    await user.click(screen.getByRole("button", { name: "Upload dataset" }));

    expect(await screen.findByText("Uploaded support_qa")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /support_qa data\/support_qa\.csv 2 rows/i })
    ).toBeInTheDocument();
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock.mock.calls.filter(([input]) => String(input) === "/api/datasets")).toHaveLength(2);
  });

  it("saves manual Q&A to the selected dataset", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: customDatasetButtonName }));
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

  it("loads reports when switching to Reports", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Reports" }));

    expect(await screen.findByText("smoke.html")).toBeInTheDocument();
  });

  it("loads evaluation runs when switching to Records", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Records" }));

    expect(await screen.findByText("smoke")).toBeInTheDocument();
  });

  it("shows run controls", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Standard evaluation" })).toBeInTheDocument());
  });
});
