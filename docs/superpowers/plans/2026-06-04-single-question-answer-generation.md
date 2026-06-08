# Single Question Answer Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-question assisted QA workflow where the user enters Q, the backend retrieves passages, the model generates A, and the user reviews/saves the confirmed QA row.

**Architecture:** Keep changes inside the existing console API module and dataset page template. Backend helpers in `rag_eval_pipeline/api.py` will read the existing pipeline config, call retrieval, extract passages, call the configured chat model, and expose two JSON endpoints. Frontend changes replace the current multi-candidate AI mode with one "Generate Answer" panel that reuses the existing manual QA save endpoint for final writes.

**Tech Stack:** Python standard library HTTP server, YAML config via existing helpers, OpenAI-compatible chat endpoint, vanilla HTML/CSS/JS, pytest static/backend tests.

---

## File Structure

- Modify `rag_eval_pipeline/api.py`
  - Add retrieval config loading and request helpers near existing chat/QA generation helpers.
  - Add `generate_answer_from_question()` and `regenerate_answer_from_passages()`.
  - Add handler methods and routes for `/api/datasets/generate-answer` and `/api/datasets/regenerate-answer`.
  - Replace the current AI-generate form markup and JS with a single-question answer-generation flow.
  - Add translation keys for the new controls and messages.
- Modify `tests/test_dataset_upload_api.py`
  - Add backend helper tests for retrieval, answer generation, regeneration, and endpoint routing.
  - Update frontend/static tests from old candidate generation UI to new Generate Answer UI.
- Modify `scripts/functional_smoke_test.py`
  - Add hook checks for `/api/datasets/generate-answer`, `/api/datasets/regenerate-answer`, and `id="generateAnswerForm"`.

---

### Task 1: Backend Helpers for Retrieval-Grounded Answer Generation

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing backend helper tests**

Add these tests near existing `test_generate_qa_candidates_*` tests in `tests/test_dataset_upload_api.py`:

```python
def test_generate_answer_from_question_calls_retrieval_and_chat(monkeypatch, tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text(
        "generation:\n"
        "  retrieval_url: http://retrieval.test/api/v1/retrieval\n"
        "  retrieval_api_key: retrieval-key\n"
        "  dataset_ids: ds-1\n"
        "  page_size: 2\n"
        "  similarity_threshold: 0.1\n",
        encoding="utf-8",
    )
    captured_retrieval = {}
    captured_messages = []

    def fake_post_json(url, payload, headers, timeout_seconds=30):
        captured_retrieval.update({"url": url, "payload": payload, "headers": headers})
        return {
            "code": 0,
            "data": {
                "chunks": [
                    {"content": "SM94 M3 is top side SR undulation.", "score": 0.91, "document_name": "defects.pdf"},
                    {"content": "It is rejected by the defect criteria.", "score": 0.83},
                ]
            },
        }

    def fake_call_chat_completion(messages, settings):
        captured_messages.extend(messages)
        return "SM94 M3 is top side solder resist undulation and is rejected by the criteria."

    monkeypatch.setattr(api, "post_retrieval_json", fake_post_json)
    monkeypatch.setattr(api, "call_chat_completion", fake_call_chat_completion)
    monkeypatch.setattr(
        api,
        "chat_settings_from_env",
        lambda: {"base_url": "http://model.test/v1", "api_key": "EMPTY", "model": "mock-chat"},
    )

    payload = api.generate_answer_from_question("What is SM94 M3?", "zh", config_path)

    assert payload["ok"] is True
    assert payload["question"] == "What is SM94 M3?"
    assert payload["expected_answer"].startswith("SM94 M3")
    assert len(payload["passages"]) == 2
    assert captured_retrieval["url"] == "http://retrieval.test/api/v1/retrieval"
    assert captured_retrieval["payload"]["question"] == "What is SM94 M3?"
    assert captured_retrieval["headers"]["Authorization"] == "Bearer retrieval-key"
    assert "retrieved passages" in captured_messages[-1]["content"].lower()


def test_generate_answer_from_question_rejects_empty_question(tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text("generation:\n  retrieval_url: http://retrieval.test\n", encoding="utf-8")

    with pytest.raises(api.UploadError) as exc:
        api.generate_answer_from_question("   ", "zh", config_path)

    assert "question" in str(exc.value).lower()


def test_generate_answer_from_question_rejects_missing_retrieval_url(tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text("generation: {}\n", encoding="utf-8")

    with pytest.raises(api.UploadError) as exc:
        api.generate_answer_from_question("What is SM94?", "zh", config_path)

    assert "retrieval" in str(exc.value).lower()


def test_generate_answer_from_question_rejects_empty_passages(monkeypatch, tmp_path):
    config_path = tmp_path / "eval-cfg.yaml"
    config_path.write_text(
        "generation:\n  retrieval_url: http://retrieval.test/api/v1/retrieval\n  retrieval_payload_json: '{\"question\":\"{query}\"}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "post_retrieval_json", lambda url, payload, headers, timeout_seconds=30: {"data": {"chunks": []}})

    with pytest.raises(api.UploadError) as exc:
        api.generate_answer_from_question("What is SM94?", "zh", config_path)

    assert "passages" in str(exc.value).lower()


def test_regenerate_answer_from_passages_calls_chat(monkeypatch):
    captured_messages = []

    def fake_call_chat_completion(messages, settings):
        captured_messages.extend(messages)
        return "Improved answer based only on the retrieved passages."

    monkeypatch.setattr(api, "call_chat_completion", fake_call_chat_completion)
    monkeypatch.setattr(
        api,
        "chat_settings_from_env",
        lambda: {"base_url": "http://model.test/v1", "api_key": "EMPTY", "model": "mock-chat"},
    )

    payload = api.regenerate_answer_from_passages(
        "What is SM94 M3?",
        "Old answer",
        [{"text": "SM94 M3 is top side SR undulation."}],
        "zh",
    )

    assert payload == {"ok": True, "expected_answer": "Improved answer based only on the retrieved passages."}
    assert "Old answer" in captured_messages[-1]["content"]


def test_regenerate_answer_from_passages_rejects_missing_passages():
    with pytest.raises(api.UploadError) as exc:
        api.regenerate_answer_from_passages("What is SM94?", "Old answer", [], "zh")

    assert "passages" in str(exc.value).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -q -k "generate_answer_from_question or regenerate_answer_from_passages"
```

Expected: FAIL because `generate_answer_from_question`, `regenerate_answer_from_passages`, and `post_retrieval_json` do not exist yet.

- [ ] **Step 3: Implement minimal backend helpers**

Add imports near the other imports in `rag_eval_pipeline/api.py`:

```python
import argparse
from rag_eval_pipeline import generation as rag_generation
```

Add helper code near `generate_qa_candidates()`:

```python
def post_retrieval_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int = 30,
) -> dict[str, Any] | list[Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise UploadError(f"Retrieval request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UploadError("Retrieval service returned invalid JSON") from exc


def retrieval_args_from_config(config_path: Path, question: str) -> argparse.Namespace:
    config = load_pipeline_config(config_path)
    generation = dict(config.get("generation") or {})
    retrieval_url = str(generation.get("retrieval_url") or "").strip()
    if not retrieval_url:
        raise UploadError("Retrieval URL is required: set generation.retrieval_url.")
    return argparse.Namespace(
        retrieval_url=retrieval_url,
        retrieval_api_key=generation.get("retrieval_api_key"),
        retrieval_auth_header=generation.get("retrieval_auth_header") or "Authorization",
        retrieval_auth_scheme=generation.get("retrieval_auth_scheme", "Bearer"),
        retrieval_headers_json=generation.get("retrieval_headers_json"),
        retrieval_payload_json=generation.get("retrieval_payload_json"),
        retrieval_extra_body_json=generation.get("retrieval_extra_body_json"),
        dataset_ids=generation.get("dataset_ids"),
        document_ids=generation.get("document_ids"),
        page=int(generation.get("page") or 1),
        page_size=int(generation.get("page_size") or 5),
        similarity_threshold=float(generation.get("similarity_threshold") or 0.0),
        vector_similarity_weight=float(generation.get("vector_similarity_weight") or 0.3),
        top_k=int(generation.get("top_k") or 1024),
        keyword=bool(generation.get("keyword", False)),
        highlight=bool(generation.get("highlight", False)),
        cross_languages=generation.get("cross_languages"),
        use_kg=bool(generation.get("use_kg", False)),
        toc_enhance=bool(generation.get("toc_enhance", False)),
        rerank_id=generation.get("rerank_id"),
        metadata_condition_json=generation.get("metadata_condition_json"),
        max_passages=int(generation.get("max_passages") or generation.get("page_size") or 5),
        question=question,
    )


def passage_payload(passages: list[rag_generation.Passage]) -> list[dict[str, Any]]:
    return [
        {
            "id": passage.passage_id,
            "text": passage.text,
            "source": passage.raw_id,
        }
        for passage in passages
    ]


def build_answer_generation_messages(question: str, passages: list[dict[str, Any]], language: str) -> list[dict[str, str]]:
    _code, prompt_language = normalize_response_language(language)
    context = "\n\n".join(
        f"[{index}] {str(passage.get('text') or '').strip()}"
        for index, passage in enumerate(passages, start=1)
        if str(passage.get("text") or "").strip()
    )
    return [
        {
            "role": "system",
            "content": (
                "You create standard answers for RAG evaluation datasets. "
                "Use only the retrieved passages. Return answer text only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Respond in {prompt_language}.\n"
                f"Question:\n{question}\n\n"
                f"Retrieved passages:\n{context}\n\n"
                "Write one factual, specific, self-contained standard answer. "
                "If the retrieved passages are insufficient, say so clearly."
            ),
        },
    ]


def generate_answer_from_question(question: str, language: str, config_path: Path = PIPELINE_CONFIG_PATH) -> dict[str, Any]:
    clean_question = question.strip()
    if not clean_question:
        raise UploadError("Question is required.")
    args = retrieval_args_from_config(config_path, clean_question)
    query = rag_generation.Query(query_id="manual_question", query=clean_question)
    retrieval_payload = rag_generation.build_retrieval_payload(args, query)
    retrieval_headers = rag_generation.build_retrieval_headers(args)
    retrieval_json = post_retrieval_json(args.retrieval_url, retrieval_payload, retrieval_headers)
    passages = rag_generation.extract_passages(retrieval_json, args.max_passages)
    if not passages:
        raise UploadError("No usable retrieved passages were found.")
    public_passages = passage_payload(passages)
    answer = call_chat_completion(
        build_answer_generation_messages(clean_question, public_passages, language),
        chat_settings_from_env(),
    ).strip()
    if not answer:
        raise UploadError("Model returned an empty answer.")
    return {"ok": True, "question": clean_question, "expected_answer": answer, "passages": public_passages}


def build_answer_regeneration_messages(
    question: str,
    current_answer: str,
    passages: list[dict[str, Any]],
    language: str,
) -> list[dict[str, str]]:
    messages = build_answer_generation_messages(question, passages, language)
    messages[-1]["content"] += (
        f"\n\nCurrent answer:\n{current_answer}\n\n"
        "Improve the current answer while staying faithful to the same retrieved passages."
    )
    return messages


def regenerate_answer_from_passages(
    question: str,
    current_answer: str,
    passages: list[dict[str, Any]],
    language: str,
) -> dict[str, Any]:
    clean_question = question.strip()
    clean_answer = current_answer.strip()
    clean_passages = [
        {"text": str(passage.get("text") or "").strip(), "source": str(passage.get("source") or "").strip()}
        for passage in passages
        if isinstance(passage, dict) and str(passage.get("text") or "").strip()
    ]
    if not clean_question:
        raise UploadError("Question is required.")
    if not clean_answer:
        raise UploadError("Current answer is required.")
    if not clean_passages:
        raise UploadError("Retrieved passages are required before regenerating.")
    answer = call_chat_completion(
        build_answer_regeneration_messages(clean_question, clean_answer, clean_passages, language),
        chat_settings_from_env(),
    ).strip()
    if not answer:
        raise UploadError("Model returned an empty answer.")
    return {"ok": True, "expected_answer": answer}
```

- [ ] **Step 4: Run helper tests to verify they pass**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -q -k "generate_answer_from_question or regenerate_answer_from_passages"
```

Expected: PASS for the new helper tests.

---

### Task 2: HTTP Routes and Handler Methods

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing route tests**

Add these tests near `test_do_post_generate_qa_returns_candidates`:

```python
def test_do_post_generate_answer_returns_answer(monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/datasets/generate-answer"
    body = json.dumps({"question": "What is SM94 M3?", "language": "zh"}).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    def fake_generate_answer_from_question(question, language, config_path):
        assert question == "What is SM94 M3?"
        assert language == "zh"
        assert config_path == handler.pipeline_config_path
        return {
            "ok": True,
            "question": question,
            "expected_answer": "Generated answer",
            "passages": [{"text": "Retrieved passage"}],
        }

    monkeypatch.setattr(api, "generate_answer_from_question", fake_generate_answer_from_question)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["expected_answer"] == "Generated answer"


def test_do_post_regenerate_answer_returns_answer(monkeypatch):
    handler = DummyUploadHandler()
    handler.path = "/api/datasets/regenerate-answer"
    body = json.dumps(
        {
            "question": "What is SM94 M3?",
            "current_answer": "Old answer",
            "passages": [{"text": "Retrieved passage"}],
            "language": "zh",
        }
    ).encode("utf-8")
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    def fake_regenerate_answer_from_passages(question, current_answer, passages, language):
        assert question == "What is SM94 M3?"
        assert current_answer == "Old answer"
        assert passages == [{"text": "Retrieved passage"}]
        assert language == "zh"
        return {"ok": True, "expected_answer": "Improved answer"}

    monkeypatch.setattr(api, "regenerate_answer_from_passages", fake_regenerate_answer_from_passages)

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert payload["expected_answer"] == "Improved answer"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_do_post_generate_answer_returns_answer tests\test_dataset_upload_api.py::test_do_post_regenerate_answer_returns_answer -q
```

Expected: FAIL because the routes are not handled yet.

- [ ] **Step 3: Add routes and handler methods**

In `DatasetUploadHandler.do_POST()`, add before `/api/datasets/generate-qa`:

```python
        if parsed_path == "/api/datasets/generate-answer":
            try:
                payload = self.handle_generate_answer()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return

        if parsed_path == "/api/datasets/regenerate-answer":
            try:
                payload = self.handle_regenerate_answer()
            except UploadError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.write_json(payload)
            return
```

Add methods near `handle_generate_qa()`:

```python
    def handle_generate_answer(self) -> dict[str, Any]:
        payload = self.read_json_body()
        return generate_answer_from_question(
            str(payload.get("question", "")),
            str(payload.get("language", "")),
            self.pipeline_config_path,
        )

    def handle_regenerate_answer(self) -> dict[str, Any]:
        payload = self.read_json_body()
        passages = payload.get("passages")
        if not isinstance(passages, list):
            raise UploadError("Retrieved passages are required before regenerating.")
        return regenerate_answer_from_passages(
            str(payload.get("question", "")),
            str(payload.get("current_answer", "")),
            passages,
            str(payload.get("language", "")),
        )
```

- [ ] **Step 4: Run route tests**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_do_post_generate_answer_returns_answer tests\test_dataset_upload_api.py::test_do_post_regenerate_answer_returns_answer -q
```

Expected: PASS.

---

### Task 3: Frontend Markup, Translations, and Static Hooks

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing frontend/static tests**

Update old candidate-generation UI assertions and add this test:

```python
def test_dataset_generate_answer_mode_has_single_question_review_flow():
    page = api.render_upload_page(result_reports=[])

    assert 'value="ai" data-i18n="generateAnswerMode"' in page
    assert 'id="aiQaPanel"' in page
    assert 'id="generateAnswerForm"' in page
    assert 'name="question"' in page
    assert 'id="generatedAnswerTextarea"' in page
    assert 'id="generateAnswerButton"' in page
    assert 'id="regenerateAnswerButton"' in page
    assert 'id="saveGeneratedAnswerButton"' in page
    assert 'id="clearGeneratedAnswerButton"' in page
    assert 'id="retrievedPassagesDetails"' in page
    assert 'id="retrievedPassagesList"' in page
    assert 'fetch("/api/datasets/generate-answer"' in page
    assert 'fetch("/api/datasets/regenerate-answer"' in page
    assert 'fetch("/api/datasets/generate-qa"' not in page
```

Also update `test_dataset_window_translation_keys_cover_all_languages` required keys:

```python
required_keys.extend([
    "generateAnswerMode",
    "generateAnswer",
    "regenerateAnswer",
    "clearGeneratedAnswer",
    "retrievedPassages",
    "generatedAnswer",
    "generatingAnswer",
    "answerGenerationFailed",
])
```

- [ ] **Step 2: Run frontend/static tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -q -k "generate_answer_mode or translation_keys_cover_all_languages or add_qa_tabs"
```

Expected: FAIL because the old AI Generate panel still uses source text, count, candidates, and `/api/datasets/generate-qa`.

- [ ] **Step 3: Replace AI panel markup**

In `render_upload_page()`, replace the `aiQaPanel` section with:

```html
                <section id="aiQaPanel" class="ai-qa-panel qa-mode-panel" data-qa-panel="ai" hidden>
                  <form id="generateAnswerForm" class="ai-qa-form answer-generation-form">
                    <label>
                      <span class="qa-field-label"><span data-i18n="question">Question</span><span class="qa-field-mark">Q</span></span>
                      <textarea name="question" placeholder="Enter one question" data-i18n-placeholder="manualQuestionPlaceholder" required></textarea>
                    </label>
                    <label>
                      <span class="qa-field-label"><span data-i18n="generatedAnswer">Generated answer</span><span class="qa-field-mark">A</span></span>
                      <textarea id="generatedAnswerTextarea" name="expected_answer" placeholder="Generated answer will appear here" required></textarea>
                    </label>
                    <div class="answer-generation-actions">
                      <button id="generateAnswerButton" type="submit" data-i18n="generateAnswer" data-ready-text="Generate answer" data-loading-text="Generating...">Generate answer</button>
                      <button id="regenerateAnswerButton" class="secondary-button" type="button" data-i18n="regenerateAnswer" disabled>Regenerate answer</button>
                      <button id="saveGeneratedAnswerButton" type="button" data-i18n="saveManualQa" disabled>Save QA</button>
                      <button id="clearGeneratedAnswerButton" class="secondary-button" type="button" data-i18n="clearGeneratedAnswer">Clear</button>
                    </div>
                  </form>
                  <details id="retrievedPassagesDetails" class="dataset-preview-details dataset-secondary-panel dataset-secondary-preview">
                    <summary class="dataset-preview-summary" data-i18n="retrievedPassages">Retrieved passages</summary>
                    <div id="retrievedPassagesList" class="retrieved-passages-list" aria-live="polite"></div>
                  </details>
                </section>
```

Change the AI option label:

```html
<option value="ai" data-i18n="generateAnswerMode">生成答案</option>
```

Keep `reportQaPanel` and `qaCandidatePanel` unchanged for report-chat candidates.

- [ ] **Step 4: Add CSS for the new panel**

Add near existing `.ai-qa-form` CSS:

```css
    .answer-generation-form {{
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      align-items: start;
    }}

    .answer-generation-actions {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: minmax(150px, 1fr) minmax(130px, 170px) minmax(130px, 170px) minmax(90px, 130px);
      gap: 10px;
      align-items: center;
    }}

    .retrieved-passages-list {{
      display: grid;
      gap: 10px;
      padding: 12px;
    }}

    .retrieved-passage-item {{
      padding: 10px 12px;
      border: 1px solid rgba(183, 204, 225, .78);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }}
```

- [ ] **Step 5: Add translation keys in all language blocks**

Add keys to `de`, `en`, `zh`, and `ms` translation blocks. English values:

```javascript
generateAnswerMode: "Generate answer",
generateAnswer: "Generate answer",
regenerateAnswer: "Regenerate",
clearGeneratedAnswer: "Clear",
retrievedPassages: "Retrieved passages",
generatedAnswer: "Generated answer",
generatingAnswer: "Generating answer...",
answerGenerationFailed: "Answer generation failed.",
```

Chinese values:

```javascript
generateAnswerMode: "生成答案",
generateAnswer: "生成答案",
regenerateAnswer: "重新生成",
clearGeneratedAnswer: "清空",
retrievedPassages: "召回片段",
generatedAnswer: "生成答案",
generatingAnswer: "正在生成答案...",
answerGenerationFailed: "答案生成失败。",
```

- [ ] **Step 6: Run frontend/static tests**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -q -k "generate_answer_mode or translation_keys_cover_all_languages or add_qa_tabs"
```

Expected: PASS after updating old assertions to the new mode names.

---

### Task 4: Frontend JavaScript for Generate and Regenerate

**Files:**
- Modify: `rag_eval_pipeline/api.py`
- Test: `tests/test_dataset_upload_api.py`

- [ ] **Step 1: Write failing JavaScript hook test**

Add:

```python
def test_dataset_generate_answer_javascript_preserves_review_before_save():
    page = api.render_upload_page(result_reports=[])

    assert 'let generatedAnswerPassages = [];' in page
    assert 'function renderRetrievedPassages(passages)' in page
    assert 'generateAnswerForm.addEventListener("submit", async (event)' in page
    assert 'regenerateAnswerButton.addEventListener("click", async ()' in page
    assert 'saveGeneratedAnswerButton.addEventListener("click", async ()' in page
    assert 'generatedAnswerTextarea.value = payload.expected_answer || "";' in page
    assert 'generatedAnswerPassages = payload.passages || [];' in page
    assert 'current_answer: generatedAnswerTextarea.value || ""' in page
    assert 'question: formData.get("question") || ""' in page
    assert 'expected_answer: generatedAnswerTextarea.value || ""' in page
```

- [ ] **Step 2: Run hook test to verify it fails**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_dataset_generate_answer_javascript_preserves_review_before_save -q
```

Expected: FAIL because the JS still manages candidate rows.

- [ ] **Step 3: Add JS constants and rendering helper**

In the main script constants section, replace old `aiQaForm`/`generateQaButton` usage with:

```javascript
    const generateAnswerForm = document.getElementById("generateAnswerForm");
    const generatedAnswerTextarea = document.getElementById("generatedAnswerTextarea");
    const generateAnswerButton = document.getElementById("generateAnswerButton");
    const regenerateAnswerButton = document.getElementById("regenerateAnswerButton");
    const saveGeneratedAnswerButton = document.getElementById("saveGeneratedAnswerButton");
    const clearGeneratedAnswerButton = document.getElementById("clearGeneratedAnswerButton");
    const retrievedPassagesList = document.getElementById("retrievedPassagesList");
    const retrievedPassagesDetails = document.getElementById("retrievedPassagesDetails");
```

Add state and helper near candidate helper functions:

```javascript
    let generatedAnswerPassages = [];

    function renderRetrievedPassages(passages) {
      if (!retrievedPassagesList) return;
      if (!passages || !passages.length) {
        retrievedPassagesList.innerHTML = "<div class=\"dataset-preview-empty\">" + escapeHTML(t("retrievedPassages")) + ": 0</div>";
        if (regenerateAnswerButton) regenerateAnswerButton.disabled = true;
        if (saveGeneratedAnswerButton) saveGeneratedAnswerButton.disabled = true;
        return;
      }
      retrievedPassagesList.innerHTML = passages.map((passage, index) => (
        "<div class=\"retrieved-passage-item\">" +
          "<strong>#" + escapeHTML(index + 1) + "</strong> " +
          escapeHTML(passage.text || "") +
          (passage.source ? "<div class=\"dataset-preview-meta\">" + escapeHTML(passage.source) + "</div>" : "") +
        "</div>"
      )).join("");
      if (retrievedPassagesDetails) retrievedPassagesDetails.open = true;
      if (regenerateAnswerButton) regenerateAnswerButton.disabled = false;
      if (saveGeneratedAnswerButton) saveGeneratedAnswerButton.disabled = !(generatedAnswerTextarea && generatedAnswerTextarea.value.trim());
    }
```

- [ ] **Step 4: Add generate/regenerate/clear event handlers**

Replace the old `aiQaForm.addEventListener("submit"... generate-qa ...)` block with:

```javascript
    if (generateAnswerForm) {
      generateAnswerForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!generateAnswerButton) return;
        generateAnswerButton.disabled = true;
        generateAnswerButton.textContent = generateAnswerButton.dataset.loadingText;
        setManualQaResult("", t("generatingAnswer"));
        try {
          const formData = new FormData(generateAnswerForm);
          const response = await fetch("/api/datasets/generate-answer", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              question: formData.get("question") || "",
              language: pageLanguage.value || "en"
            })
          });
          const payload = await response.json().catch(() => ({}));
          if (!response.ok || payload.ok === false) throw new Error(payload.error || t("answerGenerationFailed"));
          if (generatedAnswerTextarea) generatedAnswerTextarea.value = payload.expected_answer || "";
          generatedAnswerPassages = payload.passages || [];
          renderRetrievedPassages(generatedAnswerPassages);
          if (saveGeneratedAnswerButton) saveGeneratedAnswerButton.disabled = !(generatedAnswerTextarea && generatedAnswerTextarea.value.trim());
          setManualQaResult("success", escapeHTML(t("generatedAnswer")));
        } catch (error) {
          setManualQaResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
        } finally {
          generateAnswerButton.disabled = false;
          generateAnswerButton.textContent = generateAnswerButton.dataset.readyText;
        }
      });
    }

    if (regenerateAnswerButton) {
      regenerateAnswerButton.addEventListener("click", async () => {
        regenerateAnswerButton.disabled = true;
        setManualQaResult("", t("generatingAnswer"));
        try {
          const formData = new FormData(generateAnswerForm);
          const response = await fetch("/api/datasets/regenerate-answer", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              question: formData.get("question") || "",
              current_answer: generatedAnswerTextarea.value || "",
              passages: generatedAnswerPassages,
              language: pageLanguage.value || "en"
            })
          });
          const payload = await response.json().catch(() => ({}));
          if (!response.ok || payload.ok === false) throw new Error(payload.error || t("answerGenerationFailed"));
          generatedAnswerTextarea.value = payload.expected_answer || "";
          if (saveGeneratedAnswerButton) saveGeneratedAnswerButton.disabled = !(generatedAnswerTextarea && generatedAnswerTextarea.value.trim());
          setManualQaResult("success", escapeHTML(t("generatedAnswer")));
        } catch (error) {
          setManualQaResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
        } finally {
          regenerateAnswerButton.disabled = generatedAnswerPassages.length === 0;
        }
      });
    }

    if (saveGeneratedAnswerButton) {
      saveGeneratedAnswerButton.addEventListener("click", async () => {
        const formData = new FormData(generateAnswerForm);
        saveGeneratedAnswerButton.disabled = true;
        setManualQaResult("", t("manualQaSaving"));
        try {
          const response = await fetch("/api/datasets/manual-qa", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              target_dataset: manualQaDatasetSelect.value,
              question: formData.get("question") || "",
              expected_answer: generatedAnswerTextarea.value || ""
            })
          });
          const payload = await response.json().catch(() => ({}));
          if (!response.ok || payload.ok === false) throw new Error(payload.error || t("manualQaFailed"));
          setManualQaResult(
            "success",
            "<strong>" + escapeHTML(t("manualQaSaved")) + " " + escapeHTML(payload.dataset_name || "manual_qa") + "</strong>" +
              "<div>" + escapeHTML(t("rows")) + ": " + escapeHTML(payload.rows || 0) + "</div>" +
              "<div>query_id: " + escapeHTML(payload.appended_query_id || "") + "</div>" +
              "<div>" + escapeHTML(t("savedDatasetPath")) + ": " + escapeHTML(payload.path || "") + "</div>"
          );
          generateAnswerForm.reset();
          if (generatedAnswerTextarea) generatedAnswerTextarea.value = "";
          generatedAnswerPassages = [];
          renderRetrievedPassages([]);
          await refreshDatasets();
          await loadManualQaDatasetPreview();
        } catch (error) {
          setManualQaResult("error", "<strong>" + escapeHTML(t("error")) + ":</strong> " + escapeHTML(error.message));
        } finally {
          saveGeneratedAnswerButton.disabled = !(generatedAnswerTextarea && generatedAnswerTextarea.value.trim());
        }
      });
    }

    if (clearGeneratedAnswerButton) {
      clearGeneratedAnswerButton.addEventListener("click", () => {
        if (generateAnswerForm) generateAnswerForm.reset();
        if (generatedAnswerTextarea) generatedAnswerTextarea.value = "";
        generatedAnswerPassages = [];
        renderRetrievedPassages([]);
        setManualQaResult("", "");
      });
    }
```

- [ ] **Step 5: Run JS hook test and static parse**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py::test_dataset_generate_answer_javascript_preserves_review_before_save -q
@'
fetch("http://127.0.0.1:9000/datasets")
  .then((r) => r.text())
  .then((html) => {
    const re = new RegExp("<script>([\\s\\S]*?)</script>", "g");
    const scripts = Array.from(html.matchAll(re), (m) => m[1]);
    for (const script of scripts) new Function(script);
    console.log(JSON.stringify({ ok: true, scripts: scripts.length }));
  })
  .catch((error) => { console.error(error); process.exit(1); });
'@ | node
```

Expected: pytest PASS and Node script parse PASS. If local server is not running, run the Node parse after Task 5 server restart.

---

### Task 5: Verification and Smoke Test

**Files:**
- Modify: `scripts/functional_smoke_test.py`
- Test: full test suite and local API smoke.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests\test_dataset_upload_api.py -q -k "generate_answer or regenerate_answer or dataset_generate_answer"
```

Expected: all focused tests PASS.

- [ ] **Step 2: Run full tests**

Run:

```powershell
python -m py_compile rag_eval_pipeline\api.py
python -m pytest -q
```

Expected: py_compile succeeds and pytest reports all tests passing.

- [ ] **Step 3: Restart local server**

Run:

```powershell
$connections = Get-NetTCPConnection -LocalPort 9000 -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
foreach ($connection in $connections) { Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 500
Start-Process -FilePath python -ArgumentList @('-m','rag_eval_pipeline.api','--host','127.0.0.1','--port','9000') -WorkingDirectory 'c:\Users\admin\Desktop\RAG-eval\rag-eval-dev\rag-eval-dev' -RedirectStandardOutput 'server-9000.out.log' -RedirectStandardError 'server-9000.err.log' -WindowStyle Hidden -PassThru | Out-Null
Start-Sleep -Seconds 2
$listener = Get-NetTCPConnection -LocalPort 9000 -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
if (-not $listener) { Write-Error 'Server did not start on port 9000'; exit 1 }
```

Expected: port 9000 has a listening Python process.

- [ ] **Step 4: Run functional smoke**

Before running the smoke script, update its UI hook checks so the dataset page requires the new assisted-answer hooks:

```python
    required_hooks = [
        'id="manualQaForm"',
        'id="generateAnswerForm"',
        'fetch("/api/datasets/generate-answer"',
        'fetch("/api/datasets/regenerate-answer"',
        'fetch("/api/datasets/manual-qa"',
        'fetch("/api/chat"',
    ]
```

Run:

```powershell
python scripts\functional_smoke_test.py --base-url http://127.0.0.1:9000
```

Expected: smoke checks pass.

- [ ] **Step 5: Run browser-free page script parse**

Run:

```powershell
@'
fetch("http://127.0.0.1:9000/datasets")
  .then((r) => r.text())
  .then((html) => {
    const required = [
      "generateAnswerForm",
      "generatedAnswerTextarea",
      "/api/datasets/generate-answer",
      "/api/datasets/regenerate-answer",
      "retrievedPassagesList"
    ];
    const missing = required.filter((item) => !html.includes(item));
    if (missing.length) throw new Error("missing: " + missing.join(","));
    const re = new RegExp("<script>([\\s\\S]*?)</script>", "g");
    const scripts = Array.from(html.matchAll(re), (m) => m[1]);
    for (const script of scripts) new Function(script);
    console.log(JSON.stringify({ ok: true, hooks: required.length, scripts: scripts.length }));
  })
  .catch((error) => { console.error(error); process.exit(1); });
'@ | node
```

Expected: `{"ok":true,...}`.

---

## Self-Review

- Spec coverage: The plan covers single question input, retrieval, generated answer, passage display, regeneration, manual confirmation save, and error handling.
- Non-goals respected: No batch upload, no automatic writes, no scoring changes, and no pipeline execution changes are included.
- Type consistency: Backend payload keys are `question`, `expected_answer`, `passages`, and `current_answer`; frontend uses the same keys.
- Placeholder scan: No `TBD`, `TODO`, or "implement later" placeholders are intentionally left in the plan.
