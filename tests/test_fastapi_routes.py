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
