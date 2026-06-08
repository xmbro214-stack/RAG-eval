# Dataset Upload API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight local API service and browser upload page for evaluation QA CSV files.

**Architecture:** Add a focused `rag_eval_pipeline/api.py` module that owns CSV upload parsing, validation, storage, JSON responses, and the self-contained upload page. Keep the existing pipeline unchanged; users copy the returned saved CSV path into `queries_csv` and `golden_csv`.

**Tech Stack:** Python standard library HTTP server, `cgi.FieldStorage` multipart parsing, `csv`, `json`, existing `pytest` test suite, static HTML/CSS/JavaScript rendered from Python.

---

## File Structure

- Create `rag_eval_pipeline/api.py`: local HTTP service, upload validation helpers, HTML page rendering, CLI entry point.
- Create `tests/test_dataset_upload_api.py`: unit tests for slugging, validation, duplicate naming, HTML page rendering, and handler-level behavior where simple.
- Modify `README.md`: document server start, browser upload page, PowerShell upload, and config usage.
- Modify `pyproject.toml`: add console script `rag-eval-api = "rag_eval_pipeline.api:main"`.

## Task 1: Dataset CSV Validation Helpers

**Files:**
- Create: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing tests for slugging and CSV validation**

Create `tests/test_dataset_upload_api.py` with:

```python
import io

import pytest

from rag_eval_pipeline import api


def test_slugify_dataset_name_keeps_safe_name():
    assert api.slugify_dataset_name("demo qa v1.csv") == "demo_qa_v1"
    assert api.slugify_dataset_name("../bad/name") == "bad_name"
    assert api.slugify_dataset_name("   ") == "dataset"


def test_parse_and_validate_csv_normalizes_rows():
    content = "query_id,query,expected_answer,extra\n,What is A?,Answer A,ignored\nq2,What is B?,Answer B,ignored\n"

    rows = api.parse_and_validate_csv(content.encode("utf-8"))

    assert rows == [
        {"query_id": "query_1", "query": "What is A?", "expected_answer": "Answer A"},
        {"query_id": "q2", "query": "What is B?", "expected_answer": "Answer B"},
    ]


def test_parse_and_validate_csv_rejects_missing_column():
    content = "query_id,query\nq1,What is A?\n"

    with pytest.raises(api.UploadError) as exc:
        api.parse_and_validate_csv(content.encode("utf-8"))

    assert "expected_answer" in str(exc.value)


def test_parse_and_validate_csv_rejects_blank_required_values():
    content = "query_id,query,expected_answer\nq1,,Answer A\n"

    with pytest.raises(api.UploadError) as exc:
        api.parse_and_validate_csv(content.encode("utf-8"))

    assert "row 1" in str(exc.value)
    assert "query" in str(exc.value)
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -v
```

Expected: fail because `rag_eval_pipeline.api` does not exist.

- [ ] **Step 3: Implement validation helpers**

Create `rag_eval_pipeline/api.py` with:

```python
"""Lightweight API server for uploading evaluation QA CSV datasets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = ROOT / "data" / "uploaded_datasets"
REQUIRED_COLUMNS = ("query_id", "query", "expected_answer")


class UploadError(ValueError):
    """Raised when an uploaded dataset is invalid."""


def slugify_dataset_name(value: str) -> str:
    stem = Path(value.strip()).stem if value.strip() else ""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    slug = slug.strip("._-")
    return slug or "dataset"


def parse_and_validate_csv(content: bytes) -> list[dict[str, str]]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise UploadError(f"CSV missing required columns: {', '.join(missing)}")

    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=1):
        query = (row.get("query") or "").strip()
        expected_answer = (row.get("expected_answer") or "").strip()
        if not query:
            raise UploadError(f"CSV row {row_number} has blank query")
        if not expected_answer:
            raise UploadError(f"CSV row {row_number} has blank expected_answer")
        query_id = (row.get("query_id") or "").strip() or f"query_{row_number}"
        rows.append({
            "query_id": query_id,
            "query": query,
            "expected_answer": expected_answer,
        })

    if not rows:
        raise UploadError("CSV must contain at least one data row")
    return rows
```

- [ ] **Step 4: Run validation tests**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -v
```

Expected: all tests in `tests/test_dataset_upload_api.py` pass.

## Task 2: Save Uploaded Dataset Files

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Modify: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing save tests**

Append to `tests/test_dataset_upload_api.py`:

```python
def test_save_dataset_writes_normalized_csv(tmp_path):
    rows = [{"query_id": "q1", "query": "What is A?", "expected_answer": "Answer A"}]

    payload = api.save_dataset(rows, "demo", upload_root=tmp_path)

    assert payload["ok"] is True
    assert payload["dataset_name"] == "demo"
    assert payload["rows"] == 1
    assert payload["columns"] == ["query_id", "query", "expected_answer"]
    assert (tmp_path / "demo.csv").read_text(encoding="utf-8").splitlines() == [
        "query_id,query,expected_answer",
        "q1,What is A?,Answer A",
    ]


def test_save_dataset_appends_suffix_for_duplicate_names(tmp_path):
    rows = [{"query_id": "q1", "query": "What is A?", "expected_answer": "Answer A"}]
    (tmp_path / "demo.csv").write_text("existing\n", encoding="utf-8")

    payload = api.save_dataset(rows, "demo", upload_root=tmp_path)

    assert payload["dataset_name"] == "demo_2"
    assert (tmp_path / "demo_2.csv").exists()
```

- [ ] **Step 2: Run save tests to verify failure**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_save_dataset_writes_normalized_csv tests\test_dataset_upload_api.py::test_save_dataset_appends_suffix_for_duplicate_names -v
```

Expected: fail because `save_dataset` does not exist.

- [ ] **Step 3: Implement save helpers**

Append to `rag_eval_pipeline/api.py`:

```python
def unique_dataset_path(dataset_name: str, upload_root: Path = UPLOAD_ROOT) -> tuple[str, Path]:
    upload_root.mkdir(parents=True, exist_ok=True)
    base_name = slugify_dataset_name(dataset_name)
    candidate_name = base_name
    candidate_path = upload_root / f"{candidate_name}.csv"
    counter = 2
    while candidate_path.exists():
        candidate_name = f"{base_name}_{counter}"
        candidate_path = upload_root / f"{candidate_name}.csv"
        counter += 1
    return candidate_name, candidate_path


def relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def save_dataset(
    rows: list[dict[str, str]],
    dataset_name: str,
    *,
    upload_root: Path = UPLOAD_ROOT,
) -> dict[str, Any]:
    final_name, output_path = unique_dataset_path(dataset_name, upload_root)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "ok": True,
        "dataset_name": final_name,
        "path": relative_repo_path(output_path),
        "rows": len(rows),
        "columns": list(REQUIRED_COLUMNS),
    }
```

- [ ] **Step 4: Run save tests**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -v
```

Expected: all tests in `tests/test_dataset_upload_api.py` pass.

## Task 3: Upload Page HTML

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Modify: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing HTML rendering test**

Append to `tests/test_dataset_upload_api.py`:

```python
def test_render_upload_page_contains_form_and_brand():
    page = api.render_upload_page()

    assert "AT&amp;S" in page
    assert 'id="datasetUploadForm"' in page
    assert 'name="file"' in page
    assert 'name="name"' in page
    assert "/api/datasets/upload" in page
```

- [ ] **Step 2: Run HTML rendering test to verify failure**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_render_upload_page_contains_form_and_brand -v
```

Expected: fail because `render_upload_page` does not exist.

- [ ] **Step 3: Implement `render_upload_page`**

Append to `rag_eval_pipeline/api.py`:

```python
def render_upload_page() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AT&amp;S Dataset Upload</title>
  <style>
    :root {
      --bg: #07111f;
      --surface: rgba(12, 29, 52, .84);
      --ink: #f5f8ff;
      --muted: #9aa9c2;
      --line: rgba(255, 255, 255, .14);
      --blue: #63c5ff;
      --red: #ff7a66;
      --green: #42d37c;
      --field: #0b1d34;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }
    header {
      padding: 26px 32px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(7, 17, 31, .88);
      backdrop-filter: blur(18px);
    }
    .brand-lockup { display: flex; align-items: center; gap: 14px; }
    .brand-mark {
      display: grid;
      place-items: center;
      width: 62px;
      height: 42px;
      border: 1px solid rgba(99, 197, 255, .48);
      border-radius: 8px;
      background: #0d2541;
      color: white;
      font-weight: 800;
      font-size: 17px;
    }
    h1 { margin: 0 0 6px; font-size: 28px; }
    p { color: var(--muted); }
    main { padding: 24px 32px 42px; max-width: 860px; }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 18px 50px rgba(0, 0, 0, .22);
    }
    label { display: block; margin: 0 0 8px; color: var(--muted); font-size: 13px; }
    input {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      background: var(--field);
      color: var(--ink);
    }
    .field { margin-bottom: 16px; }
    button {
      height: 40px;
      border: 1px solid rgba(99, 197, 255, .48);
      border-radius: 8px;
      padding: 0 16px;
      background: #0d3155;
      color: var(--ink);
      cursor: pointer;
      font-weight: 650;
    }
    button:disabled { opacity: .58; cursor: wait; }
    .message {
      margin-top: 16px;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--line);
      overflow-wrap: anywhere;
    }
    .message.success { border-color: rgba(66, 211, 124, .45); color: var(--green); }
    .message.error { border-color: rgba(255, 122, 102, .45); color: var(--red); }
    pre {
      margin: 12px 0 0;
      padding: 12px;
      border-radius: 8px;
      background: rgba(4, 12, 24, .48);
      color: #d7e3f6;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    @media (max-width: 720px) {
      header, main { padding-left: 16px; padding-right: 16px; }
      .brand-lockup { align-items: flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand-lockup">
      <div class="brand-mark" aria-label="AT&amp;S logo">AT&amp;S</div>
      <div>
        <h1>Dataset Upload</h1>
        <p>Upload QA CSV files for RAG evaluation.</p>
      </div>
    </div>
  </header>
  <main>
    <section class="panel">
      <form id="datasetUploadForm">
        <div class="field">
          <label for="datasetFile">QA CSV file</label>
          <input id="datasetFile" name="file" type="file" accept=".csv,text/csv" required>
        </div>
        <div class="field">
          <label for="datasetName">Dataset name</label>
          <input id="datasetName" name="name" type="text" placeholder="demo">
        </div>
        <button id="uploadButton" type="submit">Upload dataset</button>
      </form>
      <div id="result" aria-live="polite"></div>
    </section>
  </main>
  <script>
    const form = document.getElementById('datasetUploadForm');
    const button = document.getElementById('uploadButton');
    const result = document.getElementById('result');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      button.textContent = 'Uploading...';
      result.innerHTML = '';
      try {
        const response = await fetch('/api/datasets/upload', {
          method: 'POST',
          body: new FormData(form)
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || 'Upload failed');
        }
        const snippet = `queries_csv: ${data.path}\\ngolden_csv: ${data.path}`;
        result.innerHTML = `
          <div class="message success">
            <strong>Uploaded ${data.dataset_name}</strong><br>
            Path: ${data.path}<br>
            Rows: ${data.rows}
            <pre>${snippet}</pre>
          </div>`;
      } catch (error) {
        result.innerHTML = `<div class="message error">${error.message}</div>`;
      } finally {
        button.disabled = false;
        button.textContent = 'Upload dataset';
      }
    });
  </script>
</body>
</html>"""
```

- [ ] **Step 4: Run HTML rendering test**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -v
```

Expected: all tests in `tests/test_dataset_upload_api.py` pass.

## Task 4: HTTP Handler And CLI Server

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Modify: `tests/test_dataset_upload_api.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing handler tests**

Append to `tests/test_dataset_upload_api.py`:

```python
from unittest.mock import Mock


class DummyUploadHandler:
    upload_root = None

    def __init__(self):
        self.status = None
        self.headers = []
        self.body = b""

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.headers.append((name, value))

    def end_headers(self):
        pass


def test_write_json_response_sets_status_and_body():
    handler = DummyUploadHandler()

    api.DatasetUploadHandler.write_json(handler, {"ok": True}, status=201)

    assert handler.status == 201
    assert ("Content-Type", "application/json; charset=utf-8") in handler.headers
    assert json.loads(handler.body.decode("utf-8")) == {"ok": True}
```

Also add `import json` at the top of `tests/test_dataset_upload_api.py`.

- [ ] **Step 2: Run handler test to verify failure**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_write_json_response_sets_status_and_body -v
```

Expected: fail because `DatasetUploadHandler` does not exist.

- [ ] **Step 3: Implement handler, multipart upload, and CLI**

Modify imports in `rag_eval_pipeline/api.py`:

```python
import cgi
```

Append to `rag_eval_pipeline/api.py`:

```python
class DatasetUploadHandler(BaseHTTPRequestHandler):
    server_version = "RagEvalDatasetAPI/0.1"
    upload_root = UPLOAD_ROOT

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/datasets":
            self.write_html(render_upload_page())
            return
        self.write_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/api/datasets/upload":
            self.write_json({"ok": False, "error": "not found"}, status=404)
            return
        try:
            payload = self.handle_upload()
        except UploadError as exc:
            self.write_json({"ok": False, "error": str(exc)}, status=400)
            return
        self.write_json(payload)

    def do_PUT(self) -> None:  # noqa: N802
        self.write_json({"ok": False, "error": "method not allowed"}, status=405)

    def handle_upload(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise UploadError("Expected multipart/form-data")
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )
        file_item = form["file"] if "file" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            raise UploadError("Missing CSV file")
        filename = str(file_item.filename)
        if not filename.lower().endswith(".csv"):
            raise UploadError("Uploaded file must be a .csv")
        content = file_item.file.read()
        name_item = form["name"] if "name" in form else None
        provided_name = ""
        if name_item is not None and not getattr(name_item, "filename", ""):
            provided_name = str(name_item.value or "").strip()
        dataset_name = provided_name or filename
        rows = parse_and_validate_csv(content)
        return save_dataset(rows, dataset_name, upload_root=self.upload_root)

    def write_html(self, html_text: str, status: int = 200) -> None:
        body = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def make_handler(upload_root: Path = UPLOAD_ROOT) -> type[DatasetUploadHandler]:
    class CustomDatasetUploadHandler(DatasetUploadHandler):
        pass

    CustomDatasetUploadHandler.upload_root = upload_root
    return CustomDatasetUploadHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--upload-root", type=Path, default=UPLOAD_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handler = make_handler(args.upload_root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Dataset upload page: http://{args.host}:{args.port}/datasets")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

Modify `pyproject.toml` under `[project.scripts]`:

```toml
rag-eval-api = "rag_eval_pipeline.api:main"
```

- [ ] **Step 4: Adjust dummy handler test for `wfile`**

Modify `DummyUploadHandler.__init__`:

```python
def __init__(self):
    self.status = None
    self.headers = []
    self.wfile = io.BytesIO()
```

Modify the assertion:

```python
assert json.loads(handler.wfile.getvalue().decode("utf-8")) == {"ok": True}
```

- [ ] **Step 5: Run handler and full upload API tests**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -v
```

Expected: all tests in `tests/test_dataset_upload_api.py` pass.

## Task 5: README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add dataset upload section**

Add a short section after the configuration section:

```markdown
## 上传评估 QA CSV

启动轻量上传服务：

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

浏览器打开：

```text
http://127.0.0.1:9000/datasets
```

上传的 CSV 至少包含：

```csv
query_id,query,expected_answer
q1,问题,标准答案
```

上传成功后页面会显示类似配置：

```yaml
queries_csv: data/uploaded_datasets/demo.csv
golden_csv: data/uploaded_datasets/demo.csv
```

把这两行填入 `scripts/eval-cfg.yaml` 后，再运行原来的 pipeline 命令。

PowerShell 也可以直接调用接口：

```powershell
curl.exe -F "file=@data\qa_golden.csv" -F "name=demo" http://127.0.0.1:9000/api/datasets/upload
```
```

- [ ] **Step 2: Add API script to common scripts list**

Add this line to the common scripts block:

```text
python -m rag_eval_pipeline.api           上传评估 QA CSV 的轻量页面和接口
```

- [ ] **Step 3: Review README rendering**

Run:

```powershell
rg -n "上传评估 QA CSV|rag_eval_pipeline.api|/datasets|api/datasets/upload" README.md
```

Expected: all four search terms are found.

## Task 6: End-To-End Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Compile Python files**

Run:

```powershell
$files = Get-ChildItem -Path rag_eval_pipeline,scripts,tests -Filter *.py -File | ForEach-Object { $_.FullName }
python -m py_compile @files
```

Expected: exit code `0`.

- [ ] **Step 2: Run test suite**

Run:

```powershell
python -m pytest
```

Expected: all tests pass.

- [ ] **Step 3: Manual server smoke test**

Start the server:

```powershell
python -m rag_eval_pipeline.api --host 127.0.0.1 --port 9000
```

In a second terminal, upload a known-good CSV:

```powershell
curl.exe -F "file=@data\qa_golden.csv" -F "name=demo" http://127.0.0.1:9000/api/datasets/upload
```

Expected response contains:

```json
"ok": true
"path": "data/uploaded_datasets/demo.csv"
```

- [ ] **Step 4: Manual browser page check**

Open:

```text
http://127.0.0.1:9000/datasets
```

Expected: the page shows the AT&S mark, a CSV file picker, dataset name field, upload button, and inline result area.

- [ ] **Step 5: Stop the manual server**

Press `Ctrl+C` in the terminal running `python -m rag_eval_pipeline.api`.

## Self-Review

- Spec coverage: API service, `GET /datasets`, `POST /api/datasets/upload`, validation, storage, duplicate naming, JSON responses, pipeline integration, tests, and README updates are covered.
- Placeholder scan: no placeholder tasks remain.
- Type consistency: helper names are consistent across tests and implementation: `slugify_dataset_name`, `parse_and_validate_csv`, `save_dataset`, `render_upload_page`, `DatasetUploadHandler`, and `make_handler`.
