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


def test_create_app_frontend_routes_fall_back_to_index_when_build_exists(tmp_path):
    static_root = tmp_path / "web" / "dist"
    static_root.mkdir(parents=True)
    (static_root / "index.html").write_text("<html><body>Console App</body></html>", encoding="utf-8")

    client = TestClient(create_app(static_root=static_root))

    for path in ["/", "/datasets", "/run", "/records", "/reports", "/overview"]:
        response = client.get(path)

        assert response.status_code == 200
        assert "Console App" in response.text


def test_legacy_api_main_runs_uvicorn_with_configured_state(monkeypatch, tmp_path):
    captured = {}
    upload_root = tmp_path / "uploads"
    reports_root = tmp_path / "reports"
    pipeline_config = tmp_path / "eval-cfg.yaml"

    def fake_run(app, host, port, reload):
        captured.update({"app": app, "host": host, "port": port, "reload": reload})

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rag-eval-api",
            "--host",
            "127.0.0.1",
            "--port",
            "9100",
            "--upload-root",
            str(upload_root),
            "--reports-root",
            str(reports_root),
            "--pipeline-config",
            str(pipeline_config),
        ],
    )

    from rag_eval_pipeline import api

    api.main()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9100
    assert captured["reload"] is False
    state = captured["app"].state.rag_eval
    assert state.upload_root == upload_root
    assert state.reports_root == reports_root
    assert state.pipeline_config_path == pipeline_config
