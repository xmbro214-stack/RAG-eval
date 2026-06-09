import { afterEach, describe, expect, it, vi } from "vitest";

import { apiClient, ApiError } from "./client";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("apiClient", () => {
  it("loads reports", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ ok: true, reports: [{ name: "smoke.html", url: "/reports/smoke.html" }] })
      })
    );

    const reports = await apiClient.listReports();

    expect(reports[0].name).toBe("smoke.html");
  });

  it("saves manual Q&A rows with backend field names", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ ok: true, appended_query_id: "query_2" })
    });
    vi.stubGlobal("fetch", fetchMock);

    const payload = await apiClient.saveManualQa("data/custom.csv", "What is A?", "Answer A");

    expect(payload.appended_query_id).toBe("query_2");
    expect(fetchMock).toHaveBeenCalledWith("/api/datasets/manual-qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_dataset: "data/custom.csv",
        question: "What is A?",
        expected_answer: "Answer A"
      })
    });
  });

  it("throws ApiError for backend errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({ ok: false, error: "Upload failed" })
      })
    );

    await expect(apiClient.listReports()).rejects.toEqual(new ApiError("Upload failed", 500));
  });
});
