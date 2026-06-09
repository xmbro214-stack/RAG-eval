import { useState } from "react";

import { apiClient } from "../api/client";
import type { PipelineJob } from "../api/types";

type RunEvaluationProps = {
  onJobStart: (job: PipelineJob) => void;
};

export function RunEvaluation({ onJobStart }: RunEvaluationProps) {
  const [error, setError] = useState("");

  async function start(mode: "quick" | "standard") {
    setError("");
    try {
      const payload = await apiClient.runPipeline(mode);
      onJobStart({ run_id: payload.run_id, status: payload.status });
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  return (
    <section className="section-panel">
      <h1>Run Evaluation</h1>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="toolbar-row">
        <button onClick={() => start("quick")} type="button">Quick evaluation</button>
        <button onClick={() => start("standard")} type="button">Standard evaluation</button>
      </div>
    </section>
  );
}
