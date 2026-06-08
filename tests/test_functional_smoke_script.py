import pytest

from scripts import mock_openai_server
from scripts.functional_smoke_test import CheckFailure, run_smoke_checks


class FakeClient:
    def __init__(self, page_html: str | None = None) -> None:
        self.page_html = page_html or """
        <html>
          <body>
            <div class="overview-summary-grid"></div>
            <div class="overview-activity-strip"></div>
            <form id="manualQaForm">
              <textarea id="manualQaQuestionTextarea"></textarea>
              <textarea id="manualQaAnswerTextarea"></textarea>
              <button id="generateAnswerButton" type="button"></button>
            </form>
            <div class="run-control-grid"></div>
            <iframe id="reportFrame"></iframe>
            <script>
              fetch("/api/datasets/generate-answer");
              fetch("/api/datasets/regenerate-answer");
              fetch("/api/datasets/manual-qa");
              fetch("/api/chat");
            </script>
          </body>
        </html>
        """
        self.uploaded = False
        self.saved_qa = False

    def get_text(self, path: str) -> str:
        assert path == "/datasets"
        return self.page_html

    def get_json(self, path: str) -> dict:
        if path == "/api/datasets":
            return {
                "ok": True,
                "datasets": [{"name": "custom", "path": "data/qa_golden.csv", "rows": 2}],
                "run_options": {"page_sizes": [5], "similarity_thresholds": [0.1]},
            }
        if path == "/api/datasets/preview?path=data%2Fqa_golden.csv":
            return {
                "ok": True,
                "rows": 2,
                "preview_rows": [{"query_id": "query_1", "query": "What is A?", "expected_answer": "Answer A"}],
            }
        if path == "/api/reports":
            return {"ok": True, "reports": []}
        if path == "/api/eval-runs":
            return {"ok": True, "runs": []}
        raise AssertionError(f"unexpected GET {path}")

    def post_multipart(self, path: str, fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> dict:
        assert path == "/api/datasets/upload"
        assert fields["name"].startswith("functional_smoke_")
        filename, content, content_type = files["file"]
        assert filename.endswith(".csv")
        assert b"query_id,query,expected_answer" in content
        assert content_type == "text/csv"
        self.uploaded = True
        return {
            "ok": True,
            "dataset_name": fields["name"],
            "path": "data/uploaded_datasets/functional_smoke_upload.csv",
            "rows": 1,
        }

    def post_json(self, path: str, payload: dict) -> dict:
        assert path == "/api/datasets/manual-qa"
        assert payload["target_dataset"] == "data/uploaded_datasets/functional_smoke_upload.csv"
        assert payload["question"]
        assert payload["expected_answer"]
        self.saved_qa = True
        return {"ok": True, "path": payload["target_dataset"], "rows": 2, "appended_query_id": "query_2"}


def test_smoke_checks_cover_read_only_console_and_api_paths():
    results = run_smoke_checks(FakeClient())

    assert [result.name for result in results] == [
        "console page",
        "datasets api",
        "dataset preview",
        "reports api",
        "eval runs api",
    ]
    assert all(result.ok for result in results)


def test_smoke_checks_fail_when_required_console_selector_is_missing():
    client = FakeClient(page_html="<html><body></body></html>")

    with pytest.raises(CheckFailure, match="overview-summary-grid"):
        run_smoke_checks(client)


def test_write_checks_upload_dataset_and_save_manual_qa():
    client = FakeClient()

    results = run_smoke_checks(client, include_write_checks=True)

    assert client.uploaded is True
    assert client.saved_qa is True
    assert [result.name for result in results][-2:] == ["dataset upload", "manual QA save"]


def test_mock_openai_server_supports_qa_generation_prompt():
    response = mock_openai_server.chat_response(
        "Generate 5 high-quality QA pairs from the source text. "
        "Return only a JSON array with objects shaped exactly as "
        '{"question":"...","expected_answer":"..."}.\n\n'
        "Source text:\nSM94 is a solder resist surface dent."
    )

    assert response.startswith("[")
    assert "question" in response
    assert "expected_answer" in response
