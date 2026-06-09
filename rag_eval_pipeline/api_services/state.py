"""Shared state for the FastAPI RAG evaluation API."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_eval_pipeline import api


@dataclass(frozen=True)
class ApiState:
    upload_root: Path = api.UPLOAD_ROOT
    reports_root: Path = api.REPORTS_ROOT
    eval_runs_root: Path = api.EVAL_RUNS_ROOT
    chat_supplement_csv_path: Path = api.CHAT_SUPPLEMENT_CSV_PATH
    pipeline_config_path: Path = api.DEFAULT_PIPELINE_CONFIG
    pipeline_jobs: dict[str, dict[str, Any]] | None = None

    def jobs(self) -> dict[str, dict[str, Any]]:
        return api.PIPELINE_JOBS if self.pipeline_jobs is None else self.pipeline_jobs
