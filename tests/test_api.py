from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, Settings
from app.main import create_app
from app.models.schemas import ChatResponse, SourceDocument
from app.services.rag_service import RAGService


def test_health_endpoint_returns_project_status(tmp_path):
    settings = Settings(
        knowledge_base_dir=PROJECT_ROOT / "data" / "knowledge",
        vector_store_dir=tmp_path / "vector_store",
        app_log_path=tmp_path / "test_app.log",
        telemetry_db_path=tmp_path / "test_telemetry.sqlite3",
        embedding_strategy="local_hash",
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["document_count"] >= 1
    assert payload["chunk_count"] >= payload["document_count"]


def test_chat_endpoint_reuses_middleware_request_id(tmp_path, monkeypatch):
    captured_request_ids: list[str | None] = []

    def fake_ask(self, payload, *, request_id=None):
        captured_request_ids.append(request_id)
        return ChatResponse(
            request_id=request_id or "generated-fallback",
            conversation_id=payload.conversation_id or "conversation-id",
            answer="测试回答",
            sources=[
                SourceDocument(
                    title="测试资料",
                    source="data/knowledge/test.md",
                    excerpt="测试摘要",
                    relevance_score=0.9,
                )
            ],
            model="test-model",
            prompt_version="test-prompt",
            latency_ms=12,
            created_at=datetime.now(UTC),
        )

    monkeypatch.setattr(RAGService, "ask", fake_ask)

    settings = Settings(
        knowledge_base_dir=PROJECT_ROOT / "data" / "knowledge",
        vector_store_dir=tmp_path / "vector_store",
        app_log_path=tmp_path / "test_app.log",
        telemetry_db_path=tmp_path / "test_telemetry.sqlite3",
        embedding_strategy="local_hash",
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/chat",
            headers={"X-Request-ID": "req-from-middleware"},
            json={"question": "仓储场景适合哪些机器人？", "conversation_id": "conv-123"},
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-from-middleware"
    assert response.json()["request_id"] == "req-from-middleware"
    assert captured_request_ids == ["req-from-middleware"]
