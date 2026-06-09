# Frontend Backend Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the current backend-rendered RAG evaluation console into a FastAPI backend and a React/Vite/TypeScript frontend under `web/`, while preserving the existing evaluation pipeline behavior.

**Architecture:** Add a FastAPI app beside the current `http.server` implementation and migrate current API behavior into focused route and service modules. Scaffold a `web/` frontend that consumes the JSON API through a small typed `fetch` client, with Vite proxying API calls during development and FastAPI serving built static assets in production.

**Tech Stack:** FastAPI, Uvicorn, Pydantic, React, Vite, TypeScript, Vitest, Testing Library, pytest.

---

## Pre-Flight Notes

- The current worktree has an uncommitted change in `rag_eval_pipeline/api.py`. Before each task that edits this file, run `git diff -- rag_eval_pipeline/api.py` and preserve user changes.
- Keep `rag_eval_pipeline/api.py` available during migration. Do not remove existing tests for the backend-rendered page until the React frontend has equivalent coverage and the compatibility entry point is updated.
- Use focused commits after each task. If a task touches both Python and frontend files, commit the smallest coherent slice.

## File Structure

Create or modify these files:

- Modify `pyproject.toml`: add FastAPI, Uvicorn, and optional dev/test dependencies.
- Modify `requirements.txt`: add runtime FastAPI dependencies.
- Create `rag_eval_pipeline/api_app.py`: FastAPI app factory, route registration, CORS for local dev, static frontend mounting, and health endpoint.
- Create `rag_eval_pipeline/api_models.py`: Pydantic request models and small response helpers.
- Create `rag_eval_pipeline/api_routes/__init__.py`: route package marker.
- Create `rag_eval_pipeline/api_routes/reports.py`: reports and eval output routes.
- Create `rag_eval_pipeline/api_routes/datasets.py`: dataset inventory, preview, upload, manual QA, answer generation, and bulk QA routes.
- Create `rag_eval_pipeline/api_routes/pipeline.py`: pipeline run, cancel, and status routes.
- Create `rag_eval_pipeline/api_routes/chat.py`: report-aware chat and chat save routes.
- Create `rag_eval_pipeline/api_services/__init__.py`: service package marker.
- Create `rag_eval_pipeline/api_services/state.py`: shared app state and dependency provider.
- Modify `rag_eval_pipeline/api.py`: delegate `main()` to FastAPI only after route parity is covered.
- Create `tests/test_fastapi_app.py`: app creation, health, static fallback, and compatibility entry tests.
- Create `tests/test_fastapi_routes.py`: FastAPI route parity tests for existing API behavior.
- Create `web/package.json`: frontend scripts and dependencies.
- Create `web/index.html`: Vite HTML entry.
- Create `web/vite.config.ts`: React plugin, API proxy, and test setup.
- Create `web/tsconfig.json`, `web/tsconfig.node.json`: TypeScript config.
- Create `web/src/main.tsx`, `web/src/App.tsx`: React entry and top-level app.
- Create `web/src/api/types.ts`, `web/src/api/client.ts`: typed frontend API client.
- Create `web/src/components/AppShell.tsx`, `web/src/components/JobRail.tsx`, `web/src/components/StatusBadge.tsx`: shared UI.
- Create `web/src/pages/Overview.tsx`, `web/src/pages/Reports.tsx`, `web/src/pages/EvaluationRuns.tsx`, `web/src/pages/RunEvaluation.tsx`, `web/src/pages/Datasets.tsx`: route-level console pages.
- Create `web/src/styles/globals.css`: console styling.
- Create `web/src/test/setup.ts`, `web/src/App.test.tsx`, `web/src/api/client.test.ts`: frontend tests.
- Modify `README.md`: document new backend and frontend development commands.

---

### Task 1: Add FastAPI App Skeleton

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `rag_eval_pipeline/api_app.py`
- Create: `rag_eval_pipeline/api_routes/__init__.py`
- Create: `rag_eval_pipeline/api_services/__init__.py`
- Create: `rag_eval_pipeline/api_services/state.py`
- Create: `tests/test_fastapi_app.py`

- [ ] **Step 1: Write failing FastAPI app tests**

Create `tests/test_fastapi_app.py`:

```python
from pathlib import Path

from fastapi.testclient import TestClient

from rag_eval_pipeline.api_app import create_app


def test_create_app_health_returns_ok(tmp_path):
    client = TestClient(create_app(static_root=tmp_path / "missing-web-dist"))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "service": "rag-eval-api"}


def test_create_app_serves_frontend_index_when_build_exists(tmp_path):
    static_root = tmp_path / "web" / "dist"
    static_root.mkdir(parents=True)
    (static_root / "index.html").write_text("<html><body>RAG Eval React</body></html>", encoding="utf-8")

    client = TestClient(create_app(static_root=static_root))

    response = client.get("/")

    assert response.status_code == 200
    assert "RAG Eval React" in response.text


def test_create_app_datasets_falls_back_to_frontend_when_build_exists(tmp_path):
    static_root = tmp_path / "web" / "dist"
    static_root.mkdir(parents=True)
    (static_root / "index.html").write_text("<html><body>Console App</body></html>", encoding="utf-8")

    client = TestClient(create_app(static_root=static_root))

    response = client.get("/datasets")

    assert response.status_code == 200
    assert "Console App" in response.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_fastapi_app.py -q
```

Expected: FAIL because `fastapi` or `rag_eval_pipeline.api_app` is not available.

- [ ] **Step 3: Add runtime dependencies**

Modify `requirements.txt`:

```text
omegaconf~=2.3.0
python-dotenv~=1.0.1
pypdf>=4.3.1
fastapi>=0.115.0,<1.0.0
uvicorn[standard]>=0.30.0,<1.0.0
python-multipart>=0.0.9,<1.0.0
```

Modify `pyproject.toml` dependencies:

```toml
dependencies = [
  "omegaconf~=2.3.0",
  "python-dotenv~=1.0.1",
  "pypdf>=4.3.1",
  "fastapi>=0.115.0,<1.0.0",
  "uvicorn[standard]>=0.30.0,<1.0.0",
  "python-multipart>=0.0.9,<1.0.0",
]
```

- [ ] **Step 4: Add shared API state**

Create `rag_eval_pipeline/api_services/state.py`:

```python
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
```

Create empty package files:

```python
# rag_eval_pipeline/api_routes/__init__.py
```

```python
# rag_eval_pipeline/api_services/__init__.py
```

- [ ] **Step 5: Add FastAPI app factory**

Create `rag_eval_pipeline/api_app.py`:

```python
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_eval_pipeline import api
from rag_eval_pipeline.api_services.state import ApiState


DEFAULT_STATIC_ROOT = api.ROOT / "web" / "dist"


def create_app(
    *,
    state: ApiState | None = None,
    static_root: Path = DEFAULT_STATIC_ROOT,
    enable_cors: bool = True,
) -> FastAPI:
    app = FastAPI(title="RAG Eval API", version="0.1.0")
    app.state.rag_eval = state or ApiState()

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "service": "rag-eval-api"}

    static_root = static_root.resolve()
    index_html = static_root / "index.html"
    if static_root.exists():
        assets_root = static_root / "assets"
        if assets_root.exists():
            app.mount("/assets", StaticFiles(directory=assets_root), name="assets")

        @app.get("/")
        @app.get("/datasets")
        def frontend_index() -> FileResponse:
            return FileResponse(index_html)

    return app


app = create_app()
```

- [ ] **Step 6: Run skeleton tests**

Run:

```powershell
python -m pytest tests/test_fastapi_app.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml requirements.txt rag_eval_pipeline/api_app.py rag_eval_pipeline/api_routes/__init__.py rag_eval_pipeline/api_services/__init__.py rag_eval_pipeline/api_services/state.py tests/test_fastapi_app.py
git commit -m "feat: add fastapi app skeleton"
```

---

### Task 2: Add Report and Dataset Read Routes

**Files:**
- Create: `rag_eval_pipeline/api_routes/reports.py`
- Create: `rag_eval_pipeline/api_routes/datasets.py`
- Modify: `rag_eval_pipeline/api_app.py`
- Create: `tests/test_fastapi_routes.py`

- [ ] **Step 1: Write failing read-route tests**

Create `tests/test_fastapi_routes.py`:

```python
import csv
import json
from pathlib import Path

from fastapi.testclient import TestClient

from rag_eval_pipeline.api_app import create_app
from rag_eval_pipeline.api_services.state import ApiState


def make_client(tmp_path, *, pipeline_jobs=None):
    upload_root = tmp_path / "uploaded"
    reports_root = tmp_path / "reports"
    eval_runs_root = tmp_path / "eval_runs"
    config_path = tmp_path / "eval-cfg.yaml"
    upload_root.mkdir()
    reports_root.mkdir()
    eval_runs_root.mkdir()
    config_path.write_text(
        "datasets:\n"
        "  - name: smoke\n"
        "    queries_csv: data/qa_golden.csv\n"
        "    golden_csv: data/qa_golden.csv\n"
        "grid:\n"
        "  page_sizes: [5]\n"
        "  similarity_thresholds: [0.1]\n",
        encoding="utf-8",
    )
    state = ApiState(
        upload_root=upload_root,
        reports_root=reports_root,
        eval_runs_root=eval_runs_root,
        chat_supplement_csv_path=tmp_path / "qa_supplement.csv",
        pipeline_config_path=config_path,
        pipeline_jobs={} if pipeline_jobs is None else pipeline_jobs,
    )
    return TestClient(create_app(state=state, static_root=tmp_path / "missing-dist"))


def write_dataset(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["query_id", "query", "expected_answer"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "query": "What is A?", "expected_answer": "Answer A"})


def test_reports_route_lists_html_reports(tmp_path):
    client = make_client(tmp_path)
    report_path = tmp_path / "reports" / "smoke.html"
    report_path.write_text("<html>report</html>", encoding="utf-8")

    response = client.get("/api/reports")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["reports"][0]["name"] == "smoke.html"
    assert payload["reports"][0]["url"] == "/reports/smoke.html"


def test_report_file_route_serves_safe_report(tmp_path):
    client = make_client(tmp_path)
    (tmp_path / "reports" / "smoke.html").write_text("<html>report</html>", encoding="utf-8")

    response = client.get("/reports/smoke.html")

    assert response.status_code == 200
    assert "report" in response.text


def test_report_file_route_rejects_path_traversal(tmp_path):
    client = make_client(tmp_path)

    response = client.get("/reports/..%2Fsecret.html")

    assert response.status_code == 404


def test_datasets_route_lists_uploaded_datasets_and_run_options(tmp_path):
    client = make_client(tmp_path)
    write_dataset(tmp_path / "uploaded" / "custom.csv")

    response = client.get("/api/datasets")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert any(dataset["name"] == "custom" for dataset in payload["datasets"])
    assert payload["run_options"]["page_sizes"] == [5]


def test_dataset_preview_route_returns_rows(tmp_path):
    client = make_client(tmp_path)
    dataset_path = tmp_path / "uploaded" / "custom.csv"
    write_dataset(dataset_path)

    response = client.get("/api/datasets/preview", params={"path": str(dataset_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["rows"] == 1
    assert payload["preview_rows"][0]["query"] == "What is A?"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_fastapi_routes.py -q
```

Expected: FAIL because route modules are not registered.

- [ ] **Step 3: Implement report routes**

Create `rag_eval_pipeline/api_routes/reports.py`:

```python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from rag_eval_pipeline import api
from rag_eval_pipeline.api_services.state import ApiState

router = APIRouter()


def get_state(request: Request) -> ApiState:
    return request.app.state.rag_eval


@router.get("/api/reports")
def list_reports(request: Request) -> dict[str, object]:
    state = get_state(request)
    return {"ok": True, "reports": api.list_result_reports(state.reports_root)}


@router.get("/reports/{report_path:path}")
def get_report(report_path: str, request: Request):
    state = get_state(request)
    safe_path = api.safe_report_path(f"/reports/{report_path}", state.reports_root)
    if safe_path is None:
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    return HTMLResponse(api.inject_report_main_button(safe_path.read_text(encoding="utf-8")))


@router.get("/eval-output/{output_path:path}")
def get_eval_output(output_path: str, request: Request):
    state = get_state(request)
    safe_path = api.safe_eval_output_path(f"/eval-output/{output_path}", state.eval_runs_root)
    if safe_path is None:
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    return PlainTextResponse(safe_path.read_text(encoding="utf-8"), media_type="text/csv")


@router.get("/api/eval-runs")
def list_eval_runs(request: Request) -> dict[str, object]:
    state = get_state(request)
    return {"ok": True, "runs": api.list_eval_runs(state.eval_runs_root, state.reports_root)}
```

- [ ] **Step 4: Implement dataset read routes**

Create `rag_eval_pipeline/api_routes/datasets.py`:

```python
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_eval_pipeline import api
from rag_eval_pipeline.api_services.state import ApiState

router = APIRouter()


def get_state(request: Request) -> ApiState:
    return request.app.state.rag_eval


def error_response(exc: api.UploadError, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=status_code)


@router.get("/api/datasets")
def list_datasets(request: Request) -> dict[str, object]:
    state = get_state(request)
    return {
        "ok": True,
        "datasets": api.list_datasets(state.upload_root, state.pipeline_config_path),
        "run_options": api.list_run_options(state.pipeline_config_path),
    }


@router.get("/api/datasets/preview")
def preview_dataset(path: str, request: Request):
    state = get_state(request)
    handler_class = api.make_handler(
        upload_root=state.upload_root,
        reports_root=state.reports_root,
        eval_runs_root=state.eval_runs_root,
        chat_supplement_csv_path=state.chat_supplement_csv_path,
        pipeline_config_path=state.pipeline_config_path,
        pipeline_jobs=state.jobs(),
    )
    try:
        dataset_path = handler_class.resolve_manual_qa_dataset_path(handler_class, path)
        return api.dataset_preview_payload(dataset_path)
    except api.UploadError as exc:
        return error_response(exc)
```

- [ ] **Step 5: Register routes in app factory**

Modify `rag_eval_pipeline/api_app.py`:

```python
from rag_eval_pipeline.api_routes import datasets, reports
```

Inside `create_app()` after CORS setup:

```python
    app.include_router(reports.router)
    app.include_router(datasets.router)
```

- [ ] **Step 6: Run read-route tests**

Run:

```powershell
python -m pytest tests/test_fastapi_routes.py -q
```

Expected: PASS for read-route tests.

- [ ] **Step 7: Commit**

```powershell
git add rag_eval_pipeline/api_app.py rag_eval_pipeline/api_routes/reports.py rag_eval_pipeline/api_routes/datasets.py tests/test_fastapi_routes.py
git commit -m "feat: add fastapi report and dataset read routes"
```

---

### Task 3: Add Dataset Write and QA Routes

**Files:**
- Modify: `rag_eval_pipeline/api_routes/datasets.py`
- Modify: `tests/test_fastapi_routes.py`

- [ ] **Step 1: Add failing dataset write-route tests**

Append to `tests/test_fastapi_routes.py`:

```python
def test_upload_route_saves_csv_dataset(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/api/datasets/upload",
        data={"name": "Support QA"},
        files={"file": ("qa.csv", b"query_id,query,expected_answer\nq1,What is A?,Answer A\n", "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["dataset_name"] == "support_qa"
    assert (tmp_path / "uploaded" / "support_qa.csv").exists()


def test_manual_qa_route_appends_row(tmp_path):
    client = make_client(tmp_path)
    dataset_path = tmp_path / "uploaded" / "custom.csv"
    write_dataset(dataset_path)

    response = client.post(
        "/api/datasets/manual-qa",
        json={
            "target_dataset": str(dataset_path),
            "question": "What is B?",
            "expected_answer": "Answer B",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["appended_query_id"] == "query_2"


def test_bulk_qa_route_appends_rows(tmp_path):
    client = make_client(tmp_path)
    dataset_path = tmp_path / "uploaded" / "custom.csv"
    write_dataset(dataset_path)

    response = client.post(
        "/api/datasets/bulk-qa",
        json={
            "target_dataset": str(dataset_path),
            "rows": [{"question": "What is C?", "expected_answer": "Answer C"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["saved"] == 1


def test_generate_qa_route_returns_candidates(monkeypatch, tmp_path):
    client = make_client(tmp_path)

    def fake_generate_qa_candidates(source_text, count, language):
        assert source_text == "SM94 text"
        assert count == 5
        assert language == "zh"
        return {"ok": True, "candidates": [{"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."}]}

    monkeypatch.setattr("rag_eval_pipeline.api.generate_qa_candidates", fake_generate_qa_candidates)

    response = client.post("/api/datasets/generate-qa", json={"source_text": "SM94 text", "count": 5, "language": "zh"})

    assert response.status_code == 200
    assert response.json()["candidates"][0]["question"] == "What is SM94?"


def test_generate_answer_route_returns_answer(monkeypatch, tmp_path):
    client = make_client(tmp_path)

    def fake_generate_answer_from_question(question, language, config_path):
        assert question == "What is SM94?"
        assert language == "zh"
        return {"ok": True, "question": question, "expected_answer": "Generated answer", "passages": []}

    monkeypatch.setattr("rag_eval_pipeline.api.generate_answer_from_question", fake_generate_answer_from_question)

    response = client.post("/api/datasets/generate-answer", json={"question": "What is SM94?", "language": "zh"})

    assert response.status_code == 200
    assert response.json()["expected_answer"] == "Generated answer"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_fastapi_routes.py -q
```

Expected: FAIL for missing dataset write routes.

- [ ] **Step 3: Add request models**

Create `rag_eval_pipeline/api_models.py`:

```python
from typing import Any

from pydantic import BaseModel


class ManualQaRequest(BaseModel):
    target_dataset: str
    question: str
    expected_answer: str


class BulkQaRequest(BaseModel):
    target_dataset: str
    rows: list[dict[str, Any]]


class GenerateQaRequest(BaseModel):
    source_text: str
    count: int = 5
    language: str = "zh"


class GenerateAnswerRequest(BaseModel):
    question: str
    language: str = "zh"


class RegenerateAnswerRequest(BaseModel):
    question: str
    current_answer: str
    passages: list[dict[str, Any]]
    language: str = "zh"
```

- [ ] **Step 4: Add dataset write routes**

Append to `rag_eval_pipeline/api_routes/datasets.py`:

```python
from fastapi import File, Form, UploadFile

from rag_eval_pipeline.api_models import (
    BulkQaRequest,
    GenerateAnswerRequest,
    GenerateQaRequest,
    ManualQaRequest,
    RegenerateAnswerRequest,
)
```

Add these route functions:

```python
async def multipart_to_saved_dataset(file: UploadFile, name: str, state: ApiState) -> dict[str, object]:
    content = await file.read()
    filename = file.filename or ""
    filename_lower = filename.lower()
    if filename_lower.endswith(".csv"):
        rows = api.parse_and_validate_csv(content)
        source_format = "csv"
    elif filename_lower.endswith(".pdf"):
        rows = api.parse_and_validate_pdf(content)
        source_format = "pdf"
    else:
        raise api.UploadError("Uploaded file must be a .csv or .pdf file")
    return api.save_dataset(rows, name.strip() or filename, upload_root=state.upload_root, source_format=source_format)


@router.post("/api/datasets/upload")
async def upload_dataset(request: Request, file: UploadFile = File(...), name: str = Form("")):
    state = get_state(request)
    try:
        return await multipart_to_saved_dataset(file, name, state)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/manual-qa")
def manual_qa(payload: ManualQaRequest, request: Request):
    state = get_state(request)
    try:
        handler_class = api.make_handler(
            upload_root=state.upload_root,
            reports_root=state.reports_root,
            eval_runs_root=state.eval_runs_root,
            chat_supplement_csv_path=state.chat_supplement_csv_path,
            pipeline_config_path=state.pipeline_config_path,
            pipeline_jobs=state.jobs(),
        )
        dataset_path = handler_class.resolve_manual_qa_dataset_path(handler_class, payload.target_dataset)
        return api.append_qa_to_dataset(dataset_path, payload.question, payload.expected_answer)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/bulk-qa")
def bulk_qa(payload: BulkQaRequest, request: Request):
    state = get_state(request)
    try:
        handler_class = api.make_handler(
            upload_root=state.upload_root,
            reports_root=state.reports_root,
            eval_runs_root=state.eval_runs_root,
            chat_supplement_csv_path=state.chat_supplement_csv_path,
            pipeline_config_path=state.pipeline_config_path,
            pipeline_jobs=state.jobs(),
        )
        dataset_path = handler_class.resolve_manual_qa_dataset_path(handler_class, payload.target_dataset)
        return api.append_bulk_qa_to_dataset(dataset_path, payload.rows)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/generate-qa")
def generate_qa(payload: GenerateQaRequest):
    try:
        return api.generate_qa_candidates(payload.source_text, payload.count, payload.language)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/generate-answer")
def generate_answer(payload: GenerateAnswerRequest, request: Request):
    state = get_state(request)
    try:
        return api.generate_answer_from_question(payload.question, payload.language, state.pipeline_config_path)
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/datasets/regenerate-answer")
def regenerate_answer(payload: RegenerateAnswerRequest, request: Request):
    state = get_state(request)
    try:
        return api.regenerate_answer_from_passages(
            payload.question,
            payload.current_answer,
            payload.passages,
            payload.language,
            state.pipeline_config_path,
        )
    except api.UploadError as exc:
        return error_response(exc)
```

- [ ] **Step 5: Run dataset route tests**

Run:

```powershell
python -m pytest tests/test_fastapi_routes.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add rag_eval_pipeline/api_models.py rag_eval_pipeline/api_routes/datasets.py tests/test_fastapi_routes.py
git commit -m "feat: add fastapi dataset write routes"
```

---

### Task 4: Add Pipeline and Chat Routes

**Files:**
- Create: `rag_eval_pipeline/api_routes/pipeline.py`
- Create: `rag_eval_pipeline/api_routes/chat.py`
- Modify: `rag_eval_pipeline/api_app.py`
- Modify: `rag_eval_pipeline/api_models.py`
- Modify: `tests/test_fastapi_routes.py`

- [ ] **Step 1: Add failing pipeline and chat tests**

Append to `tests/test_fastapi_routes.py`:

```python
def test_pipeline_status_route_returns_existing_job(tmp_path):
    jobs = {
        "run-1": {
            "run_id": "run-1",
            "status": "running",
            "log_tail": ["started"],
            "tasks": [{"status": "completed"}, {"status": "running"}],
        }
    }
    client = make_client(tmp_path, pipeline_jobs=jobs)

    response = client.get("/api/pipeline/status", params={"run_id": "run-1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["job"]["run_id"] == "run-1"
    assert payload["job"]["progress"]["total_tasks"] == 2


def test_pipeline_status_route_rejects_missing_job(tmp_path):
    client = make_client(tmp_path, pipeline_jobs={})

    response = client.get("/api/pipeline/status", params={"run_id": "missing"})

    assert response.status_code == 404
    assert response.json()["ok"] is False


def test_pipeline_run_route_delegates_to_start_job(monkeypatch, tmp_path):
    client = make_client(tmp_path)

    def fake_start_pipeline_job(config_path, stage, dry_run, overwrite, job_store):
        assert stage == "all"
        assert dry_run is False
        assert overwrite is True
        return {"ok": True, "run_id": "run-1", "status": "starting"}

    monkeypatch.setattr("rag_eval_pipeline.api.check_pipeline_retrieval_service", lambda config_path: None)
    monkeypatch.setattr("rag_eval_pipeline.api.start_pipeline_job", fake_start_pipeline_job)

    response = client.post("/api/pipeline/run", json={"mode": "standard", "stage": "all"})

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-1"


def test_chat_route_delegates_to_answer_chat_question(monkeypatch, tmp_path):
    client = make_client(tmp_path)

    def fake_answer_chat_question(question, report_url, reports_root, language):
        assert question == "What changed?"
        return {"ok": True, "answer": "Recall improved."}

    monkeypatch.setattr("rag_eval_pipeline.api.answer_chat_question", fake_answer_chat_question)

    response = client.post("/api/chat", json={"question": "What changed?", "report_url": "/reports/smoke.html", "language": "en"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Recall improved."


def test_chat_save_route_writes_supplement(tmp_path):
    client = make_client(tmp_path)

    response = client.post("/api/chat/save", json={"question": "What changed?", "answer": "Recall improved."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["supplemental_query_id"] == "query_1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_fastapi_routes.py -q
```

Expected: FAIL because pipeline and chat routes are not registered.

- [ ] **Step 3: Add pipeline and chat request models**

Append to `rag_eval_pipeline/api_models.py`:

```python
class PipelineRunRequest(BaseModel):
    mode: str = "full"
    config: str | None = None
    datasets: list[str] = []
    page_sizes: list[int] = []
    similarity_thresholds: list[float] = []
    overwrite: bool = False
    stage: str = "all"
    dry_run: bool = False


class PipelineCancelRequest(BaseModel):
    run_id: str


class ChatRequest(BaseModel):
    question: str
    report_url: str = ""
    language: str = "en"


class ChatSaveRequest(BaseModel):
    question: str
    answer: str
```

- [ ] **Step 4: Implement pipeline routes**

Create `rag_eval_pipeline/api_routes/pipeline.py`:

```python
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_eval_pipeline import api
from rag_eval_pipeline.api_models import PipelineCancelRequest, PipelineRunRequest
from rag_eval_pipeline.api_services.state import ApiState

router = APIRouter()


def get_state(request: Request) -> ApiState:
    return request.app.state.rag_eval


def error_response(exc: api.UploadError, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=status_code)


@router.get("/api/pipeline/status")
def pipeline_status(run_id: str, request: Request):
    state = get_state(request)
    job = state.jobs().get(run_id)
    if not run_id or job is None:
        return JSONResponse({"ok": False, "error": "Pipeline job not found"}, status_code=404)
    job["progress"] = api.job_progress(job)
    return {"ok": True, "job": job}


@router.post("/api/pipeline/run")
def pipeline_run(payload: PipelineRunRequest, request: Request):
    state = get_state(request)
    try:
        mode = payload.mode.strip().lower()
        config_path = api.pipeline_config_path(payload.config)
        selected_datasets = api.normalize_selected_datasets(payload.datasets)
        selected_page_sizes = api.normalize_page_sizes(payload.page_sizes)
        selected_similarity_thresholds = api.normalize_similarity_thresholds(payload.similarity_thresholds)
        overwrite = payload.overwrite
        stage = payload.stage
        dry_run = payload.dry_run
        if mode == "quick":
            config_path = api.create_quick_pipeline_config(
                config_path,
                "quick",
                selected_datasets,
                selected_page_sizes,
                selected_similarity_thresholds,
            )
            overwrite = True
        elif mode == "standard":
            config_path = api.create_standard_pipeline_config(
                config_path,
                "standard",
                selected_datasets,
                selected_page_sizes,
                selected_similarity_thresholds,
            )
            overwrite = True
        elif mode not in {"full", "all"}:
            raise api.UploadError("Pipeline mode must be full, standard, or quick")
        if api.stage_runs_generation(stage) and not dry_run:
            retrieval_error = api.check_pipeline_retrieval_service(config_path)
            if retrieval_error:
                raise api.UploadError(retrieval_error)
        return api.start_pipeline_job(
            config_path=config_path,
            stage=stage,
            dry_run=dry_run,
            overwrite=overwrite,
            job_store=state.jobs(),
        )
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/pipeline/cancel")
def pipeline_cancel(payload: PipelineCancelRequest, request: Request):
    state = get_state(request)
    try:
        return api.cancel_pipeline_job(payload.run_id, state.jobs())
    except api.UploadError as exc:
        return error_response(exc)
```

- [ ] **Step 5: Implement chat routes**

Create `rag_eval_pipeline/api_routes/chat.py`:

```python
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_eval_pipeline import api
from rag_eval_pipeline.api_models import ChatRequest, ChatSaveRequest
from rag_eval_pipeline.api_services.state import ApiState

router = APIRouter()


def get_state(request: Request) -> ApiState:
    return request.app.state.rag_eval


def error_response(exc: api.UploadError, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=status_code)


@router.post("/api/chat")
def chat(payload: ChatRequest, request: Request):
    state = get_state(request)
    try:
        if not payload.question.strip():
            raise api.UploadError("Chat question is required")
        answer_payload = api.answer_chat_question(payload.question, payload.report_url, state.reports_root, payload.language)
        return {"ok": True, **answer_payload, "reports": api.list_result_reports(state.reports_root)}
    except api.UploadError as exc:
        return error_response(exc)


@router.post("/api/chat/save")
def chat_save(payload: ChatSaveRequest, request: Request):
    state = get_state(request)
    try:
        supplement_payload = api.append_chat_supplement(payload.question, payload.answer, state.chat_supplement_csv_path)
        return {"ok": True, **supplement_payload}
    except api.UploadError as exc:
        return error_response(exc)
```

- [ ] **Step 6: Register pipeline and chat routes**

Modify `rag_eval_pipeline/api_app.py` imports:

```python
from rag_eval_pipeline.api_routes import chat, datasets, pipeline, reports
```

Add route registration:

```python
    app.include_router(pipeline.router)
    app.include_router(chat.router)
```

- [ ] **Step 7: Run FastAPI route tests**

Run:

```powershell
python -m pytest tests/test_fastapi_routes.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add rag_eval_pipeline/api_app.py rag_eval_pipeline/api_models.py rag_eval_pipeline/api_routes/pipeline.py rag_eval_pipeline/api_routes/chat.py tests/test_fastapi_routes.py
git commit -m "feat: add fastapi pipeline and chat routes"
```

---

### Task 5: Update Compatibility Entry Point

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Modify: `tests/test_fastapi_app.py`
- Modify: `README.md`

- [ ] **Step 1: Inspect current user changes**

Run:

```powershell
git diff -- rag_eval_pipeline/api.py
```

Expected: review the diff and preserve any user edits while modifying `main()`.

- [ ] **Step 2: Add failing compatibility test**

Append to `tests/test_fastapi_app.py`:

```python
def test_legacy_api_main_runs_uvicorn(monkeypatch):
    captured = {}

    def fake_run(app_path, host, port, reload):
        captured.update({"app_path": app_path, "host": host, "port": port, "reload": reload})

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["rag-eval-api", "--host", "127.0.0.1", "--port", "9100"],
    )

    from rag_eval_pipeline import api

    api.main()

    assert captured == {
        "app_path": "rag_eval_pipeline.api_app:app",
        "host": "127.0.0.1",
        "port": 9100,
        "reload": False,
    }
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_fastapi_app.py::test_legacy_api_main_runs_uvicorn -q
```

Expected: FAIL because `main()` still starts `ThreadingHTTPServer`.

- [ ] **Step 4: Update `main()` to run Uvicorn**

Modify only the bottom entry-point logic in `rag_eval_pipeline/api.py`. Keep parser options compatible.

```python
def main() -> None:
    load_dotenv_if_available()
    args = parse_args()
    import uvicorn

    uvicorn.run(
        "rag_eval_pipeline.api_app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
```

- [ ] **Step 5: Update README commands**

Modify `README.md` quick-start server command section to include:

```markdown
Start the API server:

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

During frontend development, start the React dev server from `web/` and let Vite proxy `/api` requests to the API server:

```powershell
cd web
npm install
npm run dev
```
```

- [ ] **Step 6: Run compatibility tests**

Run:

```powershell
python -m pytest tests/test_fastapi_app.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add rag_eval_pipeline/api.py tests/test_fastapi_app.py README.md
git commit -m "feat: run api entry point with fastapi"
```

---

### Task 6: Scaffold React/Vite Frontend

**Files:**
- Create: `web/package.json`
- Create: `web/index.html`
- Create: `web/vite.config.ts`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.node.json`
- Create: `web/src/test/setup.ts`
- Create: `web/src/main.tsx`
- Create: `web/src/App.tsx`
- Create: `web/src/styles/globals.css`
- Create: `web/src/App.test.tsx`

- [ ] **Step 1: Create frontend package files**

Create `web/package.json`:

```json
{
  "name": "rag-eval-web",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 127.0.0.1",
    "build": "tsc -b && vite build",
    "test": "vitest run",
    "test:watch": "vitest",
    "preview": "vite preview --host 127.0.0.1"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "vite": "^5.4.0",
    "typescript": "^5.5.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^15.0.0",
    "@testing-library/user-event": "^14.5.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "jsdom": "^24.1.0",
    "vitest": "^2.0.0"
  }
}
```

Create `web/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>RAG Evaluation Console</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Create `web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ES2020"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

Create `web/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

Create `web/vite.config.ts`:

```typescript
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:9000",
      "/reports": "http://127.0.0.1:9000",
      "/eval-output": "http://127.0.0.1:9000"
    }
  },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    globals: true
  }
});
```

- [ ] **Step 2: Add initial React app and test**

Create `web/src/test/setup.ts`:

```typescript
import "@testing-library/jest-dom/vitest";
```

Create `web/src/App.tsx`:

```tsx
import "./styles/globals.css";

export function App() {
  return (
    <main className="app-shell">
      <h1>RAG Evaluation Console</h1>
      <p>Frontend backend separation is active.</p>
    </main>
  );
}
```

Create `web/src/main.tsx`:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";

import { App } from "./App";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

Create `web/src/styles/globals.css`:

```css
:root {
  color: #172033;
  background: #eef4f8;
  font-family: Inter, "Segoe UI", Arial, sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  min-width: 320px;
  margin: 0;
}

.app-shell {
  min-height: 100vh;
  padding: 24px;
}
```

Create `web/src/App.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "./App";

describe("App", () => {
  it("renders the console title", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "RAG Evaluation Console" })).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Install frontend dependencies**

Run:

```powershell
cd web
npm install
```

Expected: `web/package-lock.json` is created and dependencies install.

- [ ] **Step 4: Run frontend tests and build**

Run:

```powershell
cd web
npm test
npm run build
```

Expected: tests PASS and Vite creates `web/dist/`.

- [ ] **Step 5: Commit**

```powershell
git add web/package.json web/package-lock.json web/index.html web/vite.config.ts web/tsconfig.json web/tsconfig.node.json web/src/test/setup.ts web/src/main.tsx web/src/App.tsx web/src/styles/globals.css web/src/App.test.tsx
git commit -m "feat: scaffold react evaluation console"
```

---

### Task 7: Add Frontend API Client

**Files:**
- Create: `web/src/api/types.ts`
- Create: `web/src/api/client.ts`
- Create: `web/src/api/client.test.ts`

- [ ] **Step 1: Write API client tests**

Create `web/src/api/client.test.ts`:

```typescript
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
        json: async () => ({ ok: false, error: "Upload failed" })
      })
    );

    await expect(apiClient.listReports()).rejects.toEqual(new ApiError("Upload failed", 500));
  });
});
```

- [ ] **Step 2: Run client tests to verify they fail**

Run:

```powershell
cd web
npm test -- src/api/client.test.ts
```

Expected: FAIL because `client.ts` does not exist.

- [ ] **Step 3: Add frontend API types**

Create `web/src/api/types.ts`:

```typescript
export type ReportItem = {
  name: string;
  path: string;
  url: string;
  modified: string;
};

export type DatasetItem = {
  name: string;
  path: string;
  rows?: number;
  source?: string;
};

export type RunOptions = {
  page_sizes: number[];
  similarity_thresholds: number[];
};

export type EvalRunItem = {
  run_name?: string;
  dataset?: string;
  page_size?: number;
  similarity_threshold?: number;
  report_url?: string;
  [key: string]: unknown;
};

export type JobProgress = {
  total_tasks?: number;
  completed_tasks?: number;
  current_task?: string;
  percent?: number;
};

export type PipelineJob = {
  run_id: string;
  status: string;
  progress?: JobProgress;
  log_tail?: string[];
  reports?: ReportItem[];
  error?: string;
};

export type DatasetPreview = {
  ok: true;
  rows: number;
  preview_rows: Array<Record<string, string>>;
};
```

- [ ] **Step 4: Add API client**

Create `web/src/api/client.ts`:

```typescript
import type { DatasetItem, DatasetPreview, EvalRunItem, PipelineJob, ReportItem, RunOptions } from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new ApiError(payload.error || response.statusText || "Request failed", response.status || 500);
  }
  return payload as T;
}

export const apiClient = {
  async listReports(): Promise<ReportItem[]> {
    const payload = await requestJson<{ ok: true; reports: ReportItem[] }>("/api/reports");
    return payload.reports;
  },

  async listEvalRuns(): Promise<EvalRunItem[]> {
    const payload = await requestJson<{ ok: true; runs: EvalRunItem[] }>("/api/eval-runs");
    return payload.runs;
  },

  async listDatasets(): Promise<{ datasets: DatasetItem[]; runOptions: RunOptions }> {
    const payload = await requestJson<{ ok: true; datasets: DatasetItem[]; run_options: RunOptions }>("/api/datasets");
    return { datasets: payload.datasets, runOptions: payload.run_options };
  },

  async previewDataset(path: string): Promise<DatasetPreview> {
    return requestJson<DatasetPreview>(`/api/datasets/preview?path=${encodeURIComponent(path)}`);
  },

  async getPipelineStatus(runId: string): Promise<PipelineJob> {
    const payload = await requestJson<{ ok: true; job: PipelineJob }>(
      `/api/pipeline/status?run_id=${encodeURIComponent(runId)}`
    );
    return payload.job;
  },

  async runPipeline(mode: "quick" | "standard"): Promise<{ run_id: string; status: string }> {
    return requestJson<{ ok: true; run_id: string; status: string }>("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, stage: "all" })
    });
  },

  async uploadDataset(name: string, file: File): Promise<Record<string, unknown>> {
    const body = new FormData();
    body.set("name", name);
    body.set("file", file);
    return requestJson<Record<string, unknown>>("/api/datasets/upload", { method: "POST", body });
  }
};
```

- [ ] **Step 5: Run frontend client tests**

Run:

```powershell
cd web
npm test -- src/api/client.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add web/src/api/types.ts web/src/api/client.ts web/src/api/client.test.ts
git commit -m "feat: add frontend api client"
```

---

### Task 8: Build Console Shell and Job Rail

**Files:**
- Modify: `web/src/App.tsx`
- Create: `web/src/components/AppShell.tsx`
- Create: `web/src/components/JobRail.tsx`
- Create: `web/src/components/StatusBadge.tsx`
- Modify: `web/src/styles/globals.css`
- Modify: `web/src/App.test.tsx`

- [ ] **Step 1: Replace app test with shell expectations**

Modify `web/src/App.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { App } from "./App";

describe("App", () => {
  it("renders console navigation and job rail", () => {
    render(<App />);

    expect(screen.getByRole("button", { name: "Datasets" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Records" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reports" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByText("Idle")).toBeInTheDocument();
  });

  it("switches between console sections", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Reports" }));

    expect(screen.getByRole("heading", { name: "Reports" })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run shell test to verify it fails**

Run:

```powershell
cd web
npm test -- src/App.test.tsx
```

Expected: FAIL because shell components are missing.

- [ ] **Step 3: Add shared components**

Create `web/src/components/StatusBadge.tsx`:

```tsx
type StatusBadgeProps = {
  status: string;
};

export function StatusBadge({ status }: StatusBadgeProps) {
  return <span className={`status-badge status-badge-${status.toLowerCase()}`}>{status}</span>;
}
```

Create `web/src/components/JobRail.tsx`:

```tsx
import type { PipelineJob } from "../api/types";
import { StatusBadge } from "./StatusBadge";

type JobRailProps = {
  job: PipelineJob | null;
};

export function JobRail({ job }: JobRailProps) {
  const status = job?.status || "Idle";
  const progress = job?.progress?.percent ?? 0;
  const latestLog = job?.log_tail?.at(-1) || "No background job is running.";

  return (
    <aside className="job-rail" aria-label="Job status">
      <div className="job-rail-header">
        <span>Job</span>
        <StatusBadge status={status} />
      </div>
      <div className="job-progress-track">
        <div className="job-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <p className="job-log-line">{latestLog}</p>
    </aside>
  );
}
```

Create `web/src/components/AppShell.tsx`:

```tsx
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
```

- [ ] **Step 4: Update App with shell state**

Modify `web/src/App.tsx`:

```tsx
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
```

- [ ] **Step 5: Add shell CSS**

Replace `web/src/styles/globals.css`:

```css
:root {
  color: #172033;
  background: #eef4f8;
  font-family: Inter, "Segoe UI", Arial, sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  min-width: 320px;
  margin: 0;
}

button,
input,
select,
textarea {
  font: inherit;
}

.app-shell {
  display: grid;
  grid-template-columns: 216px minmax(0, 1fr) 320px;
  min-height: 100vh;
}

.sidebar-nav {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 20px 14px;
  color: #fff;
  background: #004b93;
}

.brand-mark {
  margin-bottom: 12px;
  font-weight: 700;
}

.nav-button {
  min-height: 38px;
  border: 0;
  border-radius: 6px;
  color: #e9f6ff;
  background: transparent;
  text-align: left;
  cursor: pointer;
}

.nav-button.active,
.nav-button:hover {
  color: #fff;
  background: #0067bd;
}

.workspace {
  min-width: 0;
  padding: 20px;
}

.section-panel {
  min-height: calc(100vh - 40px);
  padding: 18px;
  border: 1px solid #d8e4f0;
  border-radius: 8px;
  background: #fff;
}

.job-rail {
  padding: 20px;
  border-left: 1px solid #d8e4f0;
  background: #f8fbfe;
}

.job-rail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 0 8px;
  border-radius: 6px;
  color: #124060;
  background: #dcecf7;
  font-size: 12px;
  font-weight: 700;
}

.job-progress-track {
  height: 6px;
  margin-top: 16px;
  overflow: hidden;
  border-radius: 999px;
  background: #d8e4f0;
}

.job-progress-fill {
  height: 100%;
  background: #00a7c8;
}

.job-log-line {
  color: #556575;
  font-size: 13px;
}

@media (max-width: 960px) {
  .app-shell {
    grid-template-columns: 1fr;
  }

  .sidebar-nav {
    flex-direction: row;
    overflow-x: auto;
  }

  .job-rail {
    border-top: 1px solid #d8e4f0;
    border-left: 0;
  }
}
```

- [ ] **Step 6: Run frontend shell tests**

Run:

```powershell
cd web
npm test -- src/App.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add web/src/App.tsx web/src/components/AppShell.tsx web/src/components/JobRail.tsx web/src/components/StatusBadge.tsx web/src/styles/globals.css web/src/App.test.tsx
git commit -m "feat: add react console shell"
```

---

### Task 9: Add Frontend Pages and Data Loading

**Files:**
- Create: `web/src/pages/Overview.tsx`
- Create: `web/src/pages/Reports.tsx`
- Create: `web/src/pages/EvaluationRuns.tsx`
- Create: `web/src/pages/RunEvaluation.tsx`
- Create: `web/src/pages/Datasets.tsx`
- Modify: `web/src/App.tsx`
- Modify: `web/src/api/client.ts`
- Modify: `web/src/App.test.tsx`

- [ ] **Step 1: Add page rendering tests**

Modify `web/src/App.test.tsx` to include mocked fetch responses:

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) => {
      if (path === "/api/reports") {
        return { ok: true, json: async () => ({ ok: true, reports: [{ name: "smoke.html", path: "reports/smoke.html", url: "/reports/smoke.html", modified: "2026-06-09" }] }) };
      }
      if (path === "/api/eval-runs") {
        return { ok: true, json: async () => ({ ok: true, runs: [{ run_name: "smoke", dataset: "qa", report_url: "/reports/smoke.html" }] }) };
      }
      if (path === "/api/datasets") {
        return { ok: true, json: async () => ({ ok: true, datasets: [{ name: "custom", path: "data/custom.csv", rows: 1 }], run_options: { page_sizes: [5], similarity_thresholds: [0.1] } }) };
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd web
npm test -- src/App.test.tsx
```

Expected: FAIL because pages are not implemented.

- [ ] **Step 3: Add pages**

Create `web/src/pages/Datasets.tsx`:

```tsx
import { useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { DatasetItem } from "../api/types";

export function Datasets() {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    apiClient.listDatasets().then((payload) => setDatasets(payload.datasets)).catch((exc: Error) => setError(exc.message));
  }, []);

  return (
    <section className="section-panel">
      <h1>Datasets</h1>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="toolbar-row">
        <input aria-label="Dataset name" defaultValue="" />
        <input aria-label="Dataset file" type="file" accept=".csv,.pdf,text/csv,application/pdf" />
        <button type="button">Upload</button>
      </div>
      <div className="data-table">
        {datasets.map((dataset) => (
          <div className="data-row" key={dataset.path || dataset.name}>
            <strong>{dataset.name}</strong>
            <span>{dataset.path}</span>
            <span>{dataset.rows ?? "-"}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
```

Create `web/src/pages/Reports.tsx`:

```tsx
import { useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { ReportItem } from "../api/types";

export function Reports() {
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [selectedReport, setSelectedReport] = useState<ReportItem | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiClient
      .listReports()
      .then((items) => {
        setReports(items);
        setSelectedReport(items[0] ?? null);
      })
      .catch((exc: Error) => setError(exc.message));
  }, []);

  return (
    <section className="section-panel">
      <h1>Reports</h1>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="split-panel">
        <div className="data-table">
          {reports.map((report) => (
            <button className="data-row button-row" key={report.url} onClick={() => setSelectedReport(report)} type="button">
              <strong>{report.name}</strong>
              <span>{report.modified}</span>
            </button>
          ))}
        </div>
        {selectedReport ? <iframe className="report-frame" src={selectedReport.url} title={selectedReport.name} /> : <p>No reports found.</p>}
      </div>
    </section>
  );
}
```

Create `web/src/pages/EvaluationRuns.tsx`:

```tsx
import { useEffect, useState } from "react";

import { apiClient } from "../api/client";
import type { EvalRunItem } from "../api/types";

export function EvaluationRuns() {
  const [runs, setRuns] = useState<EvalRunItem[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    apiClient.listEvalRuns().then(setRuns).catch((exc: Error) => setError(exc.message));
  }, []);

  return (
    <section className="section-panel">
      <h1>Evaluation Runs</h1>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="data-table">
        {runs.map((run, index) => (
          <div className="data-row" key={`${run.run_name || "run"}-${index}`}>
            <strong>{String(run.run_name || "-")}</strong>
            <span>{String(run.dataset || "-")}</span>
            {run.report_url ? <a href={String(run.report_url)}>Report</a> : <span>No report</span>}
          </div>
        ))}
      </div>
    </section>
  );
}
```

Create `web/src/pages/RunEvaluation.tsx`:

```tsx
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
```

Create `web/src/pages/Overview.tsx`:

```tsx
import type { PipelineJob } from "../api/types";

type OverviewProps = {
  job: PipelineJob | null;
};

export function Overview({ job }: OverviewProps) {
  return (
    <section className="section-panel">
      <h1>Overview</h1>
      <div className="summary-grid">
        <div>
          <span>Active job</span>
          <strong>{job?.status || "Idle"}</strong>
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Wire pages into App**

Modify `web/src/App.tsx`:

```tsx
import { useEffect, useState } from "react";

import type { PipelineJob } from "./api/types";
import { AppShell } from "./components/AppShell";
import { Datasets } from "./pages/Datasets";
import { EvaluationRuns } from "./pages/EvaluationRuns";
import { Overview } from "./pages/Overview";
import { Reports } from "./pages/Reports";
import { RunEvaluation } from "./pages/RunEvaluation";
import "./styles/globals.css";

type SectionId = "datasets" | "run" | "records" | "reports" | "overview";

export function App() {
  const [activeSection, setActiveSection] = useState<SectionId>("datasets");
  const [job, setJob] = useState<PipelineJob | null>(null);

  useEffect(() => {
    if (job?.run_id) {
      window.localStorage.setItem("ragEvalActiveRunId", job.run_id);
    }
  }, [job?.run_id]);

  return (
    <AppShell activeSection={activeSection} job={job} onSectionChange={setActiveSection}>
      {activeSection === "datasets" ? <Datasets /> : null}
      {activeSection === "run" ? <RunEvaluation onJobStart={setJob} /> : null}
      {activeSection === "records" ? <EvaluationRuns /> : null}
      {activeSection === "reports" ? <Reports /> : null}
      {activeSection === "overview" ? <Overview job={job} /> : null}
    </AppShell>
  );
}
```

- [ ] **Step 5: Add page CSS**

Append to `web/src/styles/globals.css`:

```css
.toolbar-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 12px 0 18px;
}

.toolbar-row input,
.toolbar-row button {
  min-height: 36px;
  border: 1px solid #c9d7e5;
  border-radius: 6px;
}

.toolbar-row button {
  padding: 0 12px;
  color: #fff;
  background: #005bac;
  cursor: pointer;
}

.data-table {
  display: grid;
  gap: 8px;
}

.data-row {
  display: grid;
  grid-template-columns: minmax(140px, 1fr) minmax(160px, 2fr) auto;
  gap: 12px;
  align-items: center;
  min-height: 40px;
  padding: 8px 10px;
  border: 1px solid #d8e4f0;
  border-radius: 6px;
  background: #f8fbfe;
  text-align: left;
}

.button-row {
  width: 100%;
  color: inherit;
  cursor: pointer;
}

.split-panel {
  display: grid;
  grid-template-columns: minmax(220px, 320px) minmax(0, 1fr);
  gap: 16px;
}

.report-frame {
  width: 100%;
  min-height: 560px;
  border: 1px solid #d8e4f0;
  border-radius: 6px;
  background: #fff;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}

.summary-grid > div {
  display: grid;
  gap: 6px;
  padding: 12px;
  border: 1px solid #d8e4f0;
  border-radius: 6px;
  background: #f8fbfe;
}

.error-text {
  color: #a83232;
}
```

- [ ] **Step 6: Run frontend tests**

Run:

```powershell
cd web
npm test
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add web/src/pages/Overview.tsx web/src/pages/Reports.tsx web/src/pages/EvaluationRuns.tsx web/src/pages/RunEvaluation.tsx web/src/pages/Datasets.tsx web/src/App.tsx web/src/api/client.ts web/src/App.test.tsx web/src/styles/globals.css
git commit -m "feat: add react console pages"
```

---

### Task 10: Final Verification and Documentation

**Files:**
- Modify: `README.md`
- Modify: `.gitignore` if `web/dist/` or frontend cache files are not ignored.

- [ ] **Step 1: Update README with split-stack commands**

Add a concise section to `README.md`:

```markdown
## Frontend and Backend Development

Start the FastAPI backend:

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

Start the React/Vite frontend:

```powershell
cd web
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Build the frontend for FastAPI static serving:

```powershell
cd web
npm run build
```

After the build, the backend serves the React app from `web/dist` at `/` and `/datasets`.
```

- [ ] **Step 2: Ensure frontend generated files are ignored**

Check `.gitignore` for these entries:

```gitignore
web/node_modules/
web/dist/
web/.vite/
```

Add them if missing.

- [ ] **Step 3: Run backend tests**

Run:

```powershell
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 4: Run frontend tests and build**

Run:

```powershell
cd web
npm test
npm run build
```

Expected: PASS and `web/dist/` is generated.

- [ ] **Step 5: Verify static frontend serving**

Run backend server:

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

In another terminal:

```powershell
curl.exe http://127.0.0.1:9000/api/health
curl.exe http://127.0.0.1:9000/datasets
```

Expected: `/api/health` returns `{"ok":true,"service":"rag-eval-api"}` and `/datasets` returns the built React HTML.

- [ ] **Step 6: Manual browser check**

Open:

```text
http://127.0.0.1:5173
```

Verify:

- Datasets page loads dataset inventory.
- Reports page lists HTML reports and previews a selected report.
- Records page lists evaluation runs.
- Run page has quick and standard evaluation buttons.
- Job rail remains visible.
- Mobile viewport stacks navigation and job rail without overlapping text.

- [ ] **Step 7: Commit**

```powershell
git add README.md .gitignore
git commit -m "docs: document split frontend backend workflow"
```

---

## Self-Review Checklist

- Spec coverage: backend FastAPI app, route modules, frontend `web/`, Vite proxy, static serving, API client, console pages, error display, and testing are each mapped to tasks.
- Scope: this plan keeps local file storage and excludes authentication, database persistence, distributed queues, and pipeline rewrites.
- Compatibility: the plan keeps `api.py` until FastAPI route parity is tested, then delegates `main()` to Uvicorn.
- Type consistency: frontend `PipelineJob`, `ReportItem`, `DatasetItem`, and route response names match the API client usage.
- Verification: backend tests, frontend tests, frontend build, static serving, and manual browser checks are included.
