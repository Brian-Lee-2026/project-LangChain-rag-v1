from app.core.config import Settings
from app.services.rag_service import RAGService


def test_rag_service_reuses_persisted_chroma_index(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "warehouse.md").write_text(
        "# 仓储场景\n仓储物流适合优先自动化与拣选辅助。",
        encoding="utf-8",
    )
    settings = Settings(
        knowledge_base_dir=knowledge_dir,
        vector_store_dir=tmp_path / "vector_store",
        app_log_path=tmp_path / "app.log",
        telemetry_db_path=tmp_path / "telemetry.sqlite3",
        embedding_strategy="local_hash",
        vector_store_collection_name="test_collection",
    )
    settings.ensure_directories()

    first = RAGService(settings, force_rebuild=True)
    second = RAGService(settings)

    assert first.index_status == "rebuilt"
    assert second.index_status == "loaded"
    assert settings.vector_store_manifest_path.exists()
    assert second._retrieve_context(question="仓储自动化", top_k=2)


def test_rag_service_rebuilds_persisted_index_when_knowledge_changes(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    document_path = knowledge_dir / "warehouse.md"
    document_path.write_text("# 仓储场景\n仓储物流适合优先自动化。", encoding="utf-8")
    settings = Settings(
        knowledge_base_dir=knowledge_dir,
        vector_store_dir=tmp_path / "vector_store",
        app_log_path=tmp_path / "app.log",
        telemetry_db_path=tmp_path / "telemetry.sqlite3",
        embedding_strategy="local_hash",
        vector_store_collection_name="test_collection",
    )
    settings.ensure_directories()

    RAGService(settings, force_rebuild=True)
    document_path.write_text(
        "# 仓储场景\n仓储物流适合优先自动化，并支持动态补货。",
        encoding="utf-8",
    )

    rebuilt = RAGService(settings)

    assert rebuilt.index_status == "rebuilt"
