from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, Settings
from app.main import create_app


def test_health_endpoint_returns_project_status(tmp_path):
    settings = Settings(
        knowledge_base_dir=PROJECT_ROOT / "data" / "knowledge",
        vector_store_dir=tmp_path / "vector_store",
        feedback_log_path=tmp_path / "test_feedback.jsonl",
        rag_audit_path=tmp_path / "test_rag_events.jsonl",
        app_log_path=tmp_path / "test_app.log",
        embedding_strategy="local_hash",
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["document_count"] >= 1
    assert payload["chunk_count"] >= payload["document_count"]
