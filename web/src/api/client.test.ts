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
