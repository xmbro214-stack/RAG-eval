import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) => {
      if (path === "/api/reports") {
        return {
          ok: true,
          json: async () => ({
            ok: true,
            reports: [
              {
                name: "smoke.html",
                path: "reports/smoke.html",
                url: "/reports/smoke.html",
                modified: "2026-06-09"
              }
            ]
          })
        };
      }
      if (path === "/api/eval-runs") {
        return {
          ok: true,
          json: async () => ({ ok: true, runs: [{ run_name: "smoke", dataset: "qa", report_url: "/reports/smoke.html" }] })
        };
      }
      if (path === "/api/datasets") {
        return {
          ok: true,
          json: async () => ({
            ok: true,
            datasets: [{ name: "custom", path: "data/custom.csv", rows: 1 }],
            run_options: { page_sizes: [5], similarity_thresholds: [0.1] }
          })
        };
      }
      return { ok: true, json: async () => ({ ok: true }) };
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
