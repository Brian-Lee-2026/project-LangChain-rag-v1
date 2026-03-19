import math

import pytest

from app.core.config import Settings
from app.services.embeddings import LocalHashEmbeddings, build_embeddings


def test_local_hash_embeddings_are_deterministic():
    embeddings = LocalHashEmbeddings(size=64)

    first = embeddings.embed_query("具身机器人在仓储场景中的需求")
    second = embeddings.embed_query("具身机器人在仓储场景中的需求")

    assert first == second
    assert len(first) == 64


def test_local_hash_embeddings_are_normalized():
    embeddings = LocalHashEmbeddings(size=64)
    vector = embeddings.embed_query("制造业柔性装配")
    norm = math.sqrt(sum(value * value for value in vector))

    assert math.isclose(norm, 1.0, rel_tol=1e-6)


def test_huggingface_embeddings_fallback_to_local_hash(monkeypatch, tmp_path):
    module = pytest.importorskip("langchain_huggingface")

    class BrokenEmbeddings:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("hf download failed")

    monkeypatch.setattr(module, "HuggingFaceEmbeddings", BrokenEmbeddings)
    settings = Settings(
        knowledge_base_dir=tmp_path / "kb",
        vector_store_dir=tmp_path / "vector",
        embedding_cache_dir=tmp_path / "model_cache",
        app_log_path=tmp_path / "app.log",
        telemetry_db_path=tmp_path / "telemetry.sqlite3",
        embedding_strategy="huggingface",
        embedding_fallback_to_local_hash=True,
    )
    settings.ensure_directories()

    embeddings = build_embeddings(settings)

    assert isinstance(embeddings, LocalHashEmbeddings)
