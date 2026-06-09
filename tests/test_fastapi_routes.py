import csv
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
        writer.writerow({"query_id": "query_1", "query": "What is A?", "expected_answer": "Answer A"})


def test_reports_route_lists_html_reports(tmp_path):
    client = make_client(tmp_path)
    report_path = tmp_path / "reports" / "smoke.html"
    report_path.write_text("<html>report</html>", encoding="utf-8")

    response = client.get("/api/reports")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["reports"][0]["name"] == "smoke.html"
    assert payload["reports"][0]["url"].startswith("/reports/smoke.html?v=")


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


def test_regenerate_answer_route_returns_answer(monkeypatch, tmp_path):
    client = make_client(tmp_path)

    def fake_regenerate_answer_from_passages(question, current_answer, passages, language, config_path):
        assert question == "What is SM94?"
        assert current_answer == "Old answer"
        assert passages == [{"text": "Retrieved passage"}]
        assert language == "zh"
        return {"ok": True, "expected_answer": "Improved answer"}

    monkeypatch.setattr("rag_eval_pipeline.api.regenerate_answer_from_passages", fake_regenerate_answer_from_passages)

    response = client.post(
        "/api/datasets/regenerate-answer",
        json={
            "question": "What is SM94?",
            "current_answer": "Old answer",
            "passages": [{"text": "Retrieved passage"}],
            "language": "zh",
        },
    )

    assert response.status_code == 200
    assert response.json()["expected_answer"] == "Improved answer"


def test_retrieval_chunks_route_returns_chunks(monkeypatch, tmp_path):
    client = make_client(tmp_path)

    def fake_retrieve_chunks_from_question(
        question,
        config_path,
        *,
        page_size=None,
        similarity_threshold=None,
        dataset_ids=None,
        document_ids=None,
        max_passages=None,
    ):
        assert question == "What is SM94?"
        assert config_path.name == "eval-cfg.yaml"
        assert page_size == 2
        assert similarity_threshold == 0.2
        assert dataset_ids == ["ds-1"]
        assert document_ids == ["doc-1"]
        assert max_passages == 1
        return {
            "ok": True,
            "question": question,
            "chunks": [{"id": "[1]", "text": "Retrieved chunk", "source": "chunk-1"}],
            "retrieval": {"page_size": 2, "similarity_threshold": 0.2},
        }

    monkeypatch.setattr("rag_eval_pipeline.api.retrieve_chunks_from_question", fake_retrieve_chunks_from_question)

    response = client.post(
        "/api/retrieval/chunks",
        json={
            "question": "What is SM94?",
            "page_size": 2,
            "similarity_threshold": 0.2,
            "dataset_ids": ["ds-1"],
            "document_ids": ["doc-1"],
            "max_passages": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["chunks"][0]["text"] == "Retrieved chunk"


def test_pipeline_status_route_returns_existing_job(tmp_path):
    jobs = {
        "run-1": {
            "run_id": "run-1",
            "status": "running",
            "log_tail": ["started"],
            "expected_tasks": 2,
            "entries": [{"evaluation_status": "completed"}, {"evaluation_status": "running"}],
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


def test_pipeline_cancel_route_delegates_to_cancel_job(monkeypatch, tmp_path):
    client = make_client(tmp_path, pipeline_jobs={"run-1": {"run_id": "run-1", "status": "running"}})

    def fake_cancel_pipeline_job(run_id, job_store):
        assert run_id == "run-1"
        assert "run-1" in job_store
        return {"ok": True, "run_id": run_id, "status": "cancelling"}

    monkeypatch.setattr("rag_eval_pipeline.api.cancel_pipeline_job", fake_cancel_pipeline_job)

    response = client.post("/api/pipeline/cancel", json={"run_id": "run-1"})

    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"


def test_chat_route_delegates_to_answer_chat_question(monkeypatch, tmp_path):
    client = make_client(tmp_path)

    def fake_answer_chat_question(question, report_url, reports_root, language):
        assert question == "What changed?"
        assert report_url == "/reports/smoke.html"
        assert language == "en"
        return {"answer": "Recall improved."}

    monkeypatch.setattr("rag_eval_pipeline.api.answer_chat_question", fake_answer_chat_question)

    response = client.post(
        "/api/chat",
        json={"question": "What changed?", "report_url": "/reports/smoke.html", "language": "en"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Recall improved."


def test_chat_save_route_writes_supplement(tmp_path):
    client = make_client(tmp_path)

    response = client.post("/api/chat/save", json={"question": "What changed?", "answer": "Recall improved."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["supplemental_query_id"] == "query_1"
