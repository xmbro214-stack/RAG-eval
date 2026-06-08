# AI-Assisted QA Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AI-assisted QA generation and report-chat-to-QA candidate workflows to reduce manual dataset authoring.

**Architecture:** Extend the existing single-file dataset console API in `rag_eval_pipeline/api.py` with candidate generation and bulk-save helpers, then add an in-memory candidate review UI to the Datasets section. Keep all model-generated QA as candidates until the user explicitly saves selected rows.

**Tech Stack:** Python standard library HTTP server, CSV, JSON, inline HTML/CSS/JavaScript in `rag_eval_pipeline/api.py`, pytest tests in `tests/test_dataset_upload_api.py`.

---

## File Structure

- Modify `rag_eval_pipeline/api.py`
  - Add backend helpers for QA candidate parsing, model prompting, and bulk append.
  - Add `POST /api/datasets/generate-qa` and `POST /api/datasets/bulk-qa`.
  - Update `render_upload_page()` markup, CSS, i18n, and JavaScript for Add QA tabs and candidate review.
  - Add report-chat "Add to QA Candidates" action.
- Modify `tests/test_dataset_upload_api.py`
  - Add backend helper and endpoint tests.
  - Add static page tests for the new tabs and buttons.
  - Add report-chat candidate action test.
- Optionally modify `scripts/functional_smoke_test.py`
  - Only if existing smoke checks fail because required UI hooks changed.

Do not split `api.py` in this implementation. It is large, but the existing project pattern keeps the dataset server, page renderer, and tests together.

---

### Task 1: Backend Bulk QA Append

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing tests for bulk QA append**

Add these tests near `test_handle_manual_qa_appends_row_to_target_dataset` in `tests/test_dataset_upload_api.py`:

```python
def test_append_bulk_qa_to_dataset_appends_multiple_rows(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text(
        "query_id,query,expected_answer\n"
        "query_1,Existing question,Existing answer\n",
        encoding="utf-8",
    )

    payload = api.append_bulk_qa_to_dataset(
        target_csv,
        [
            {"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."},
            {"question": "What is SM94 M3?", "expected_answer": "SM94 M3 is top-side SR undulation."},
        ],
    )

    assert payload["ok"] is True
    assert payload["dataset_name"] == "support_qa"
    assert payload["source_format"] == "manual_bulk"
    assert payload["saved"] == 2
    assert payload["appended_query_ids"] == ["query_2", "query_3"]
    assert payload["rows"] == 3
    assert target_csv.read_text(encoding="utf-8").splitlines() == [
        "query_id,query,expected_answer",
        "query_1,Existing question,Existing answer",
        "query_2,What is SM94?,SM94 is an SR dent.",
        "query_3,What is SM94 M3?,SM94 M3 is top-side SR undulation.",
    ]


def test_append_bulk_qa_to_dataset_rejects_empty_rows(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text("query_id,query,expected_answer\n", encoding="utf-8")

    with pytest.raises(api.UploadError) as exc:
        api.append_bulk_qa_to_dataset(target_csv, [])

    assert "at least one QA row" in str(exc.value)


def test_append_bulk_qa_to_dataset_rejects_blank_question_or_answer(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text("query_id,query,expected_answer\n", encoding="utf-8")

    with pytest.raises(api.UploadError) as exc:
        api.append_bulk_qa_to_dataset(
            target_csv,
            [{"question": "What is SM94?", "expected_answer": "   "}],
        )

    assert "question and standard answer" in str(exc.value)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_append_bulk_qa_to_dataset_appends_multiple_rows tests\test_dataset_upload_api.py::test_append_bulk_qa_to_dataset_rejects_empty_rows tests\test_dataset_upload_api.py::test_append_bulk_qa_to_dataset_rejects_blank_question_or_answer -q
```

Expected: fail because `append_bulk_qa_to_dataset` does not exist.

- [ ] **Step 3: Implement bulk append helper**

Add this after `append_qa_to_dataset()` in `rag_eval_pipeline/api.py`:

```python
def append_bulk_qa_to_dataset(dataset_csv_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise UploadError("Save requires at least one QA row")

    existing_rows = read_dataset_rows(dataset_csv_path)
    next_id_number = 1
    for row in existing_rows:
        match = re.fullmatch(r"query_(\d+)", (row.get("query_id") or "").strip())
        if match:
            next_id_number = max(next_id_number, int(match.group(1)) + 1)

    clean_rows: list[dict[str, str]] = []
    appended_query_ids: list[str] = []
    for item in rows:
        question = str(item.get("question", "")).strip()
        expected_answer = str(item.get("expected_answer", "")).strip()
        if not question or not expected_answer:
            raise UploadError("Each QA row requires a question and standard answer")
        query_id = f"query_{next_id_number}"
        next_id_number += 1
        appended_query_ids.append(query_id)
        clean_rows.append({
            "query_id": query_id,
            "query": question,
            "expected_answer": expected_answer,
        })

    with dataset_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(REQUIRED_COLUMNS))
        if dataset_csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(clean_rows)

    return {
        "ok": True,
        "dataset_name": dataset_csv_path.stem,
        "path": relative_repo_path(dataset_csv_path),
        "rows": len(existing_rows) + len(clean_rows),
        "saved": len(clean_rows),
        "appended_query_ids": appended_query_ids,
        "columns": list(REQUIRED_COLUMNS),
        "source_format": "manual_bulk",
    }
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_append_bulk_qa_to_dataset_appends_multiple_rows tests\test_dataset_upload_api.py::test_append_bulk_qa_to_dataset_rejects_empty_rows tests\test_dataset_upload_api.py::test_append_bulk_qa_to_dataset_rejects_blank_question_or_answer -q
```

Expected: all pass.

- [ ] **Step 5: Commit if git identity is configured**

Run:

```powershell
git add rag_eval_pipeline\api.py tests\test_dataset_upload_api.py
git commit -m "feat: add bulk qa append helper"
```

Expected: commit succeeds. If git identity is not configured, skip commit and report that the files remain uncommitted.

---

### Task 2: Backend AI Candidate Generation

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing tests for candidate parsing and generation**

Add these tests near chat helper tests in `tests/test_dataset_upload_api.py`:

```python
def test_parse_qa_candidates_accepts_json_array():
    candidates = api.parse_qa_candidates(
        '[{"question":"What is SM94?","expected_answer":"SM94 is an SR dent."}]'
    )

    assert candidates == [
        {"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."}
    ]


def test_parse_qa_candidates_rejects_invalid_json():
    with pytest.raises(api.UploadError) as exc:
        api.parse_qa_candidates("not json")

    assert "Candidate QA could not be parsed" in str(exc.value)


def test_generate_qa_candidates_calls_chat_model(monkeypatch):
    captured_messages = []

    def fake_call_chat_completion(messages, settings):
        captured_messages.extend(messages)
        return json.dumps([
            {"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."}
        ])

    monkeypatch.setattr(api, "call_chat_completion", fake_call_chat_completion)
    monkeypatch.setattr(
        api,
        "chat_settings_from_env",
        lambda: {"base_url": "http://model.test/v1", "api_key": "EMPTY", "model": "mock-chat"},
    )

    payload = api.generate_qa_candidates("SM94 is a solder resist dent.", 5, "zh")

    assert payload["ok"] is True
    assert payload["candidates"][0]["question"] == "What is SM94?"
    assert "Return only a JSON array" in captured_messages[-1]["content"]


def test_generate_qa_candidates_rejects_empty_source_text():
    with pytest.raises(api.UploadError) as exc:
        api.generate_qa_candidates("   ", 5, "zh")

    assert "source text" in str(exc.value)


def test_generate_qa_candidates_rejects_invalid_count():
    with pytest.raises(api.UploadError) as exc:
        api.generate_qa_candidates("SM94 text", 7, "zh")

    assert "count" in str(exc.value)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_parse_qa_candidates_accepts_json_array tests\test_dataset_upload_api.py::test_parse_qa_candidates_rejects_invalid_json tests\test_dataset_upload_api.py::test_generate_qa_candidates_calls_chat_model tests\test_dataset_upload_api.py::test_generate_qa_candidates_rejects_empty_source_text tests\test_dataset_upload_api.py::test_generate_qa_candidates_rejects_invalid_count -q
```

Expected: fail because helper functions do not exist.

- [ ] **Step 3: Implement candidate parsing and generation helpers**

Add this near `answer_chat_question()` helpers in `rag_eval_pipeline/api.py`:

```python
QA_GENERATION_COUNTS = {5, 10, 20}


def parse_qa_candidates(text: str) -> list[dict[str, str]]:
    clean_text = text.strip()
    try:
        parsed = json.loads(clean_text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", clean_text, flags=re.DOTALL)
        if not match:
            raise UploadError("Candidate QA could not be parsed. Regenerate or edit manually.")
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise UploadError("Candidate QA could not be parsed. Regenerate or edit manually.") from exc
    if not isinstance(parsed, list):
        raise UploadError("Candidate QA response must be a JSON array")

    candidates: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        expected_answer = str(item.get("expected_answer", "")).strip()
        if question and expected_answer:
            candidates.append({"question": question, "expected_answer": expected_answer})
    if not candidates:
        raise UploadError("Candidate QA response did not include usable rows")
    return candidates


def build_qa_generation_messages(source_text: str, count: int, language: str) -> list[dict[str, str]]:
    _code, prompt_language = normalize_response_language(language)
    return [
        {
            "role": "system",
            "content": (
                "You create QA pairs for RAG evaluation datasets. "
                "Return strict JSON only. Do not include markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate {count} high-quality QA pairs from the source text. "
                f"Use {prompt_language} for the question when appropriate. "
                "Each question must be answerable from the source text. "
                "Each expected_answer must be factual, specific, and self-contained. "
                "Return only a JSON array with objects shaped exactly as "
                "{\"question\":\"...\",\"expected_answer\":\"...\"}.\n\n"
                f"Source text:\n{source_text}"
            ),
        },
    ]


def generate_qa_candidates(source_text: str, count: int, language: str = "zh") -> dict[str, Any]:
    clean_source_text = source_text.strip()
    if not clean_source_text:
        raise UploadError("Please provide source text.")
    if count not in QA_GENERATION_COUNTS:
        raise UploadError("QA generation count must be one of 5, 10, or 20")
    messages = build_qa_generation_messages(clean_source_text, count, language)
    response_text = call_chat_completion(messages, chat_settings_from_env())
    candidates = parse_qa_candidates(response_text)
    return {"ok": True, "candidates": candidates[:count]}
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_parse_qa_candidates_accepts_json_array tests\test_dataset_upload_api.py::test_parse_qa_candidates_rejects_invalid_json tests\test_dataset_upload_api.py::test_generate_qa_candidates_calls_chat_model tests\test_dataset_upload_api.py::test_generate_qa_candidates_rejects_empty_source_text tests\test_dataset_upload_api.py::test_generate_qa_candidates_rejects_invalid_count -q
```

Expected: all pass.

- [ ] **Step 5: Commit if git identity is configured**

Run:

```powershell
git add rag_eval_pipeline\api.py tests\test_dataset_upload_api.py
git commit -m "feat: generate qa candidates from source text"
```

Expected: commit succeeds. If git identity is not configured, skip commit and report that the files remain uncommitted.

---

### Task 3: Add API Endpoints

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing endpoint tests**

Add these tests near `test_do_post_manual_qa_saves_dataset`:

```python
def test_do_post_generate_qa_returns_candidates(monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/datasets/generate-qa"
    body = json.dumps({"source_text": "SM94 text", "count": 5, "language": "zh"}).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    def fake_generate_qa_candidates(source_text, count, language):
        assert source_text == "SM94 text"
        assert count == 5
        assert language == "zh"
        return {"ok": True, "candidates": [{"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."}]}

    monkeypatch.setattr(api, "generate_qa_candidates", fake_generate_qa_candidates)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert payload["candidates"][0]["question"] == "What is SM94?"


def test_do_post_bulk_qa_saves_selected_rows(tmp_path):
    target_csv = tmp_path / "support_qa.csv"
    target_csv.write_text("query_id,query,expected_answer\n", encoding="utf-8")
    handler = DummyUploadHandler()
    handler.upload_root = tmp_path
    handler.path = "/api/datasets/bulk-qa"
    body = json.dumps(
        {
            "target_dataset": str(target_csv),
            "rows": [{"question": "What is SM94?", "expected_answer": "SM94 is an SR dent."}],
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["ok"] is True
    assert payload["saved"] == 1
    assert payload["appended_query_ids"] == ["query_1"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_do_post_generate_qa_returns_candidates tests\test_dataset_upload_api.py::test_do_post_bulk_qa_saves_selected_rows -q
```

Expected: fail because routes and handlers do not exist.

- [ ] **Step 3: Add handler methods**

Add these methods to `DatasetUploadHandler` after `handle_manual_qa()`:

```python
def handle_generate_qa(self) -> dict[str, Any]:
    payload = self.read_json_body()
    return generate_qa_candidates(
        str(payload.get("source_text", "")),
        int(payload.get("count") or 5),
        str(payload.get("language", "")),
    )


def handle_bulk_qa(self) -> dict[str, Any]:
    payload = self.read_json_body()
    dataset_csv_path = self.resolve_manual_qa_dataset_path(str(payload.get("target_dataset", "")))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise UploadError("QA rows must be a list")
    return append_bulk_qa_to_dataset(dataset_csv_path, rows)
```

- [ ] **Step 4: Route new endpoints in `do_POST()`**

In `DatasetUploadHandler.do_POST()`, add these branches before the upload fallback:

```python
if parsed_path == "/api/datasets/generate-qa":
    try:
        payload = self.handle_generate_qa()
    except UploadError as exc:
        self.write_json({"ok": False, "error": str(exc)}, status=400)
        return
    self.write_json(payload)
    return

if parsed_path == "/api/datasets/bulk-qa":
    try:
        payload = self.handle_bulk_qa()
    except UploadError as exc:
        self.write_json({"ok": False, "error": str(exc)}, status=400)
        return
    self.write_json(payload)
    return
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_do_post_generate_qa_returns_candidates tests\test_dataset_upload_api.py::test_do_post_bulk_qa_saves_selected_rows -q
```

Expected: all pass.

- [ ] **Step 6: Commit if git identity is configured**

Run:

```powershell
git add rag_eval_pipeline\api.py tests\test_dataset_upload_api.py
git commit -m "feat: add qa generation api endpoints"
```

Expected: commit succeeds. If git identity is not configured, skip commit and report that the files remain uncommitted.

---

### Task 4: Add Datasets Page Candidate Review UI

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing static UI tests**

Add these tests near the existing render page tests:

```python
def test_render_upload_page_contains_add_qa_tabs():
    page = api.render_upload_page(result_reports=[])

    assert 'id="qaModeManual"' in page
    assert 'id="qaModeAiGenerate"' in page
    assert 'id="qaModeReportChat"' in page
    assert 'id="aiQaSourceText"' in page
    assert 'id="generateQaButton"' in page
    assert 'id="qaCandidateTable"' in page
    assert 'id="saveSelectedQaButton"' in page


def test_render_upload_page_calls_generate_and_bulk_qa_endpoints():
    page = api.render_upload_page(result_reports=[])

    assert 'fetch("/api/datasets/generate-qa"' in page
    assert 'fetch("/api/datasets/bulk-qa"' in page
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_render_upload_page_contains_add_qa_tabs tests\test_dataset_upload_api.py::test_render_upload_page_calls_generate_and_bulk_qa_endpoints -q
```

Expected: fail because new UI hooks do not exist.

- [ ] **Step 3: Update Datasets markup**

In `render_upload_page()` replace the current manual QA panel body with this structure while preserving the existing target dataset select, path label, preview, and manual form fields:

```html
<div class="manual-qa-panel">
  <h3 class="section-heading" data-i18n="addQa">Add QA</h3>
  <div class="qa-mode-tabs" role="tablist" aria-label="Add QA mode">
    <button id="qaModeManual" class="qa-mode-tab active" type="button" data-qa-mode="manual" data-i18n="manualAdd">Manual</button>
    <button id="qaModeAiGenerate" class="qa-mode-tab" type="button" data-qa-mode="ai" data-i18n="aiGenerate">AI Generate</button>
    <button id="qaModeReportChat" class="qa-mode-tab" type="button" data-qa-mode="report" data-i18n="fromReportChat">From Report Chat</button>
  </div>
  <form id="manualQaForm" class="manual-qa-form">
    <!-- keep existing target dataset, preview, question, expected_answer fields -->
  </form>
  <section id="aiQaPanel" class="qa-mode-panel" hidden>
    <label>
      <span data-i18n="sourceText">Source text</span>
      <textarea id="aiQaSourceText" rows="7" data-i18n-placeholder="sourceTextPlaceholder" placeholder="Paste report or defect text"></textarea>
    </label>
    <label>
      <span data-i18n="candidateCount">Candidate count</span>
      <select id="aiQaCount">
        <option value="5">5</option>
        <option value="10" selected>10</option>
        <option value="20">20</option>
      </select>
    </label>
    <div class="manual-qa-actions">
      <button id="generateQaButton" type="button" data-i18n="generateCandidateQa">Generate Candidate QA</button>
      <button id="clearQaCandidatesButton" type="button" data-i18n="clearCandidates">Clear Candidates</button>
    </div>
  </section>
  <section id="reportQaPanel" class="qa-mode-panel" hidden>
    <p class="section-note" data-i18n="reportQaHint">Use Add to QA Candidates from the report question panel.</p>
  </section>
  <section id="qaCandidatePanel" class="qa-candidate-panel">
    <div class="dataset-preview-title">
      <span data-i18n="candidateQa">Candidate QA</span>
      <span id="qaCandidateCount" class="dataset-preview-meta">0</span>
    </div>
    <div class="table-scroll">
      <table id="qaCandidateTable" class="records-table">
        <thead>
          <tr>
            <th data-i18n="select">Select</th>
            <th data-i18n="question">Question</th>
            <th data-i18n="standardAnswer">Standard answer</th>
            <th data-i18n="actions">Actions</th>
          </tr>
        </thead>
        <tbody id="qaCandidateRows"></tbody>
      </table>
    </div>
    <div class="manual-qa-actions">
      <button id="saveSelectedQaButton" type="button" data-i18n="saveSelectedQa">Save Selected QA</button>
    </div>
  </section>
  <div id="manualQaResult" class="result" role="status" aria-live="polite"></div>
</div>
```

Keep the original manual form IDs:

- `manualQaDatasetSelect`
- `manualQaDatasetPath`
- `manualQaDatasetPreview`
- `manualQaForm`

- [ ] **Step 4: Add CSS for tabs and candidate table**

Add CSS near the existing manual QA styles:

```css
.qa-mode-tabs {
  display: inline-flex;
  gap: 6px;
  padding: 4px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f8fbff;
}
.qa-mode-tab {
  min-height: 34px;
  padding: 0 12px;
  border: 0;
  color: var(--ats-blue-dark);
  background: transparent;
}
.qa-mode-tab.active {
  color: #fff;
  background: var(--ats-blue);
}
.qa-mode-panel {
  display: grid;
  gap: 12px;
}
.qa-candidate-panel {
  display: grid;
  gap: 12px;
  min-width: 0;
}
.qa-candidate-textarea {
  min-height: 72px;
  resize: vertical;
}
```

- [ ] **Step 5: Add i18n entries**

In each language object, add at least these keys. Chinese values:

```javascript
addQa: "添加 QA",
manualAdd: "手动添加",
aiGenerate: "AI 生成",
fromReportChat: "从报告问答加入",
sourceText: "来源文本",
sourceTextPlaceholder: "粘贴报告、缺陷说明或工艺描述",
candidateCount: "候选数量",
generateCandidateQa: "生成候选 QA",
clearCandidates: "清空候选",
candidateQa: "候选 QA",
saveSelectedQa: "保存选中 QA",
select: "选择",
actions: "操作",
reportQaHint: "在报告问答区域点击加入 QA 候选区。",
noCandidates: "暂无候选 QA。",
generatingQa: "正在生成候选 QA...",
qaGenerationFailed: "候选 QA 生成失败。",
selectQaRows: "请至少选择一条 QA。",
selectedQaSaved: "已保存选中 QA"
```

Use clear English equivalents for the English language object. German and Malay can use concise English fallback if existing translations are inconsistent.

- [ ] **Step 6: Add candidate JavaScript state and rendering**

Add after existing DOM lookups:

```javascript
const qaModeTabs = Array.from(document.querySelectorAll(".qa-mode-tab"));
const aiQaPanel = document.getElementById("aiQaPanel");
const reportQaPanel = document.getElementById("reportQaPanel");
const aiQaSourceText = document.getElementById("aiQaSourceText");
const aiQaCount = document.getElementById("aiQaCount");
const generateQaButton = document.getElementById("generateQaButton");
const clearQaCandidatesButton = document.getElementById("clearQaCandidatesButton");
const qaCandidateRows = document.getElementById("qaCandidateRows");
const qaCandidateCount = document.getElementById("qaCandidateCount");
const saveSelectedQaButton = document.getElementById("saveSelectedQaButton");
let qaCandidates = [];
```

Add these functions:

```javascript
function setQaMode(mode) {
  qaModeTabs.forEach((button) => button.classList.toggle("active", button.dataset.qaMode === mode));
  if (manualQaForm) manualQaForm.hidden = mode !== "manual";
  if (aiQaPanel) aiQaPanel.hidden = mode !== "ai";
  if (reportQaPanel) reportQaPanel.hidden = mode !== "report";
}

function renderQaCandidates() {
  if (!qaCandidateRows) return;
  qaCandidateCount.textContent = String(qaCandidates.length);
  if (!qaCandidates.length) {
    qaCandidateRows.innerHTML = "<tr><td colspan=\"4\">" + escapeHTML(t("noCandidates")) + "</td></tr>";
    return;
  }
  qaCandidateRows.innerHTML = qaCandidates.map((candidate, index) =>
    "<tr data-index=\"" + index + "\">" +
      "<td><input type=\"checkbox\" class=\"qa-candidate-select\" " + (candidate.selected ? "checked" : "") + "></td>" +
      "<td><textarea class=\"qa-candidate-question qa-candidate-textarea\">" + escapeHTML(candidate.question || "") + "</textarea></td>" +
      "<td><textarea class=\"qa-candidate-answer qa-candidate-textarea\">" + escapeHTML(candidate.expected_answer || "") + "</textarea></td>" +
      "<td><button type=\"button\" class=\"qa-candidate-delete\">" + escapeHTML(t("delete")) + "</button></td>" +
    "</tr>"
  ).join("");
}

function syncQaCandidateRow(row) {
  const index = Number(row.dataset.index);
  if (!Number.isInteger(index) || !qaCandidates[index]) return;
  qaCandidates[index].selected = row.querySelector(".qa-candidate-select")?.checked || false;
  qaCandidates[index].question = row.querySelector(".qa-candidate-question")?.value || "";
  qaCandidates[index].expected_answer = row.querySelector(".qa-candidate-answer")?.value || "";
}

function addQaCandidates(rows) {
  rows.forEach((row) => {
    qaCandidates.push({
      selected: true,
      question: row.question || "",
      expected_answer: row.expected_answer || ""
    });
  });
  renderQaCandidates();
}
```

Add listeners:

```javascript
qaModeTabs.forEach((button) => button.addEventListener("click", () => setQaMode(button.dataset.qaMode || "manual")));

qaCandidateRows.addEventListener("input", (event) => {
  const row = event.target.closest("tr[data-index]");
  if (row) syncQaCandidateRow(row);
});
qaCandidateRows.addEventListener("change", (event) => {
  const row = event.target.closest("tr[data-index]");
  if (row) syncQaCandidateRow(row);
});
qaCandidateRows.addEventListener("click", (event) => {
  if (!event.target.classList.contains("qa-candidate-delete")) return;
  const row = event.target.closest("tr[data-index]");
  const index = Number(row.dataset.index);
  qaCandidates.splice(index, 1);
  renderQaCandidates();
});
clearQaCandidatesButton.addEventListener("click", () => {
  qaCandidates = [];
  renderQaCandidates();
});
```

- [ ] **Step 7: Add generate and save actions**

Add:

```javascript
generateQaButton.addEventListener("click", async () => {
  generateQaButton.disabled = true;
  setManualQaResult("", t("generatingQa"));
  try {
    const response = await fetch("/api/datasets/generate-qa", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        source_text: aiQaSourceText.value || "",
        count: Number(aiQaCount.value || 10),
        language: pageLanguage.value || "zh"
      })
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) throw new Error(payload.error || t("qaGenerationFailed"));
    addQaCandidates(payload.candidates || []);
    setManualQaResult("success", escapeHTML(t("candidateQa")) + ": " + escapeHTML((payload.candidates || []).length));
  } catch (error) {
    setManualQaResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
  } finally {
    generateQaButton.disabled = false;
  }
});

saveSelectedQaButton.addEventListener("click", async () => {
  document.querySelectorAll("#qaCandidateRows tr[data-index]").forEach(syncQaCandidateRow);
  const selectedRows = qaCandidates
    .filter((candidate) => candidate.selected)
    .map((candidate) => ({
      question: candidate.question.trim(),
      expected_answer: candidate.expected_answer.trim()
    }));
  if (!selectedRows.length) {
    setManualQaResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(t("selectQaRows")));
    return;
  }
  const targetDataset = manualQaDatasetSelect.value || "";
  saveSelectedQaButton.disabled = true;
  setManualQaResult("", t("manualQaSaving"));
  try {
    const response = await fetch("/api/datasets/bulk-qa", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({target_dataset: targetDataset, rows: selectedRows})
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) throw new Error(payload.error || t("manualQaFailed"));
    setManualQaResult(
      "success",
      "<strong>" + escapeHTML(t("selectedQaSaved")) + "</strong>" +
        "<div>" + escapeHTML(t("rows")) + ": " + escapeHTML(payload.rows || 0) + "</div>" +
        "<div>" + escapeHTML(t("savedDatasetPath")) + ": " + escapeHTML(payload.path || "") + "</div>"
    );
    qaCandidates = qaCandidates.filter((candidate) => !candidate.selected);
    renderQaCandidates();
    await refreshDatasets();
    await loadManualQaDatasetPreview();
  } catch (error) {
    setManualQaResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
  } finally {
    saveSelectedQaButton.disabled = false;
  }
});

setQaMode("manual");
renderQaCandidates();
```

- [ ] **Step 8: Run static UI tests and syntax checks**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_render_upload_page_contains_add_qa_tabs tests\test_dataset_upload_api.py::test_render_upload_page_calls_generate_and_bulk_qa_endpoints -q
node -e "fetch('http://127.0.0.1:9000/datasets').then(r=>r.text()).then(html=>{const scripts=Array.from(html.matchAll(/<script>([\\s\\S]*?)<\\/script>/g),m=>m[1]); for (const s of scripts) new Function(s); console.log('scripts ok', scripts.length);})"
```

Expected: pytest passes. Node script parse passes if the server is running with the new code; if not, run the Python compile and functional smoke after restarting the server.

- [ ] **Step 9: Commit if git identity is configured**

Run:

```powershell
git add rag_eval_pipeline\api.py tests\test_dataset_upload_api.py
git commit -m "feat: add qa candidate review ui"
```

Expected: commit succeeds. If git identity is not configured, skip commit and report that the files remain uncommitted.

---

### Task 5: Add Report Chat to QA Candidates

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing static test**

Add this test near report section tests:

```python
def test_report_chat_can_add_answer_to_qa_candidates():
    page = api.render_upload_page(result_reports=[])

    assert 'id="addReportAnswerToQaButton"' in page
    assert "addQaCandidates([{question:" in page
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_report_chat_can_add_answer_to_qa_candidates -q
```

Expected: fail because the button and logic do not exist.

- [ ] **Step 3: Add report-chat button**

In the report chat markup, add this button near the ask-model button or below the answer result:

```html
<button id="addReportAnswerToQaButton" type="button" data-i18n="addToQaCandidates" disabled>Add to QA Candidates</button>
```

Add i18n:

```javascript
addToQaCandidates: "加入 QA 候选区",
addedToQaCandidates: "已加入 QA 候选区"
```

Use clear English equivalents in the English object.

- [ ] **Step 4: Store last report chat question and answer**

Add DOM lookup:

```javascript
const addReportAnswerToQaButton = document.getElementById("addReportAnswerToQaButton");
let lastReportQuestion = "";
let lastReportAnswer = "";
```

In the successful `/api/chat` branch:

```javascript
lastReportQuestion = String(question || "");
lastReportAnswer = String(payload.answer || "");
if (addReportAnswerToQaButton) addReportAnswerToQaButton.disabled = !(lastReportQuestion && lastReportAnswer);
```

In the error branch:

```javascript
lastReportQuestion = "";
lastReportAnswer = "";
if (addReportAnswerToQaButton) addReportAnswerToQaButton.disabled = true;
```

- [ ] **Step 5: Add button behavior**

Add:

```javascript
if (addReportAnswerToQaButton) {
  addReportAnswerToQaButton.addEventListener("click", () => {
    if (!lastReportQuestion || !lastReportAnswer) return;
    addQaCandidates([{question: lastReportQuestion, expected_answer: lastReportAnswer}]);
    setQaMode("report");
    setManualQaResult("success", escapeHTML(t("addedToQaCandidates")));
  });
}
```

- [ ] **Step 6: Run test and smoke**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_report_chat_can_add_answer_to_qa_candidates -q
python scripts\functional_smoke_test.py --base-url http://127.0.0.1:9000
```

Expected: static test passes. Smoke test passes after server restart if needed.

- [ ] **Step 7: Commit if git identity is configured**

Run:

```powershell
git add rag_eval_pipeline\api.py tests\test_dataset_upload_api.py
git commit -m "feat: add report chat qa candidate action"
```

Expected: commit succeeds. If git identity is not configured, skip commit and report that the files remain uncommitted.

---

### Task 6: Full Verification and Server Restart

**Files:**
- Verify: `rag_eval_pipeline/api.py`
- Verify: `tests/test_dataset_upload_api.py`
- Verify: `scripts/functional_smoke_test.py`

- [ ] **Step 1: Run Python compile check**

Run:

```powershell
python -m py_compile rag_eval_pipeline\api.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Restart local server**

Run:

```powershell
$connections = Get-NetTCPConnection -LocalPort 9000 -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
foreach ($c in $connections) { Stop-Process -Id $c.OwningProcess -Force }
Start-Sleep -Seconds 1
Start-Process -FilePath python -ArgumentList @('-m','rag_eval_pipeline.api','--host','127.0.0.1','--port','9000') -WorkingDirectory 'c:\Users\admin\Desktop\RAG-eval\rag-eval-dev\rag-eval-dev' -RedirectStandardOutput 'server-9000.out.log' -RedirectStandardError 'server-9000.err.log' -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 2
Get-NetTCPConnection -LocalPort 9000 -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' } | Select-Object LocalAddress,LocalPort,OwningProcess,State
```

Expected: port 9000 is listening.

- [ ] **Step 4: Run functional smoke**

Run:

```powershell
python scripts\functional_smoke_test.py --base-url http://127.0.0.1:9000
```

Expected: all smoke checks pass.

- [ ] **Step 5: Manual browser check**

Open:

```text
http://127.0.0.1:9000/datasets
```

Verify:

- Datasets section shows Add QA tabs.
- Manual tab still saves one QA row.
- AI Generate tab can generate mocked or real candidates when chat env is configured.
- Candidate table supports edit, delete, select.
- Save Selected QA appends rows and refreshes preview.
- Report chat answer can be added to QA candidates.

- [ ] **Step 6: Final status**

Report:

- Tests run and pass/fail counts.
- Server URL.
- Any skipped commit because git identity is not configured.
- Whether AI generation requires `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`, and optionally `LOCAL_LLM_API_KEY`.
