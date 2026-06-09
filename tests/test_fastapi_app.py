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
