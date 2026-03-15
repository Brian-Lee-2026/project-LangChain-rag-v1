import hashlib
import json
import os
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_chroma import Chroma

from app.core.config import Settings
from app.core.logging import append_jsonl, get_logger
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, SourceDocument
from app.services.embeddings import build_embeddings
from app.services.knowledge_base import load_knowledge_documents, split_documents
from app.services.prompting import PROMPT_VERSION, build_chat_messages

INDEX_SCHEMA_VERSION = "1.0.0"
CHROMA_COLLECTION_CONFIGURATION = {"hnsw": {"space": "cosine"}}


class RAGServiceError(RuntimeError):
    """当 RAG 服务无法完成请求时抛出。"""


class RAGService:
    def __init__(self, settings: Settings, *, force_rebuild: bool = False) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self.embeddings = build_embeddings(settings)
        self.documents = load_knowledge_documents(settings.knowledge_base_dir)
        self.chunks = split_documents(
            self.documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.index_manifest = self._build_index_manifest()
        self.index_status = "loaded"
        self.vector_store = self._load_or_build_vector_store(force_rebuild=force_rebuild)
        self._llm = None

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=self.settings.app_name,
            environment=self.settings.app_env,
            document_count=len(self.documents),
            chunk_count=len(self.chunks),
            model=self.settings.deepseek_model,
            prompt_version=PROMPT_VERSION,
        )

    def ask(self, payload: ChatRequest) -> ChatResponse:
        question = payload.question.strip()
        if not question:
            raise ValueError("问题不能为空。")

        request_id = str(uuid4())
        conversation_id = payload.conversation_id or str(uuid4())
        top_k = payload.top_k or self.settings.retrieval_k
        created_at = datetime.now(UTC)
        started_at = perf_counter()

        try:
            context_docs = self._retrieve_context(question=question, top_k=top_k)
            messages = build_chat_messages(
                question=question,
                history=payload.history[-self.settings.max_history_turns :],
                context_docs=context_docs,
            )
            answer = self._invoke_llm(messages)
            latency_ms = int((perf_counter() - started_at) * 1000)
            sources = [
                SourceDocument(
                    title=item["title"],
                    source=item["source"],
                    excerpt=_clip_text(item["content"], 140),
                    relevance_score=min(max(item["score"], 0.0), 1.0),
                )
                for item in context_docs
            ]
            append_jsonl(
                self.settings.rag_audit_path,
                {
                    "event": "rag_answer",
                    "environment": self.settings.app_env,
                    "request_id": request_id,
                    "conversation_id": conversation_id,
                    "question": question,
                    "answer": answer,
                    "model": self.settings.deepseek_model,
                    "prompt_version": PROMPT_VERSION,
                    "latency_ms": latency_ms,
                    "sources": sources_to_json(context_docs),
                    "created_at": created_at.isoformat(),
                },
            )
            self.logger.info(
                "RAG answer generated",
                extra={
                    "request_id": request_id,
                    "conversation_id": conversation_id,
                    "latency_ms": latency_ms,
                    "source_count": len(sources),
                    "environment": self.settings.app_env,
                },
            )
            return ChatResponse(
                request_id=request_id,
                conversation_id=conversation_id,
                answer=answer,
                sources=sources,
                model=self.settings.deepseek_model,
                prompt_version=PROMPT_VERSION,
                latency_ms=latency_ms,
                created_at=created_at,
            )
        except Exception as exc:
            self.logger.exception(
                "RAG answer failed",
                extra={"request_id": request_id, "question": question},
            )
            append_jsonl(
                self.settings.rag_audit_path,
                {
                    "event": "rag_error",
                    "environment": self.settings.app_env,
                    "request_id": request_id,
                    "conversation_id": conversation_id,
                    "question": question,
                    "error": str(exc),
                    "created_at": created_at.isoformat(),
                },
            )
            if isinstance(exc, RAGServiceError):
                raise
            raise RAGServiceError("问答链路执行失败，请检查日志或模型配置。") from exc

    def _invoke_llm(self, messages: list[Any]) -> str:
        llm = self._get_llm()
        result = llm.invoke(messages)
        content = getattr(result, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                (
                    block.get("text", "")
                    if isinstance(block, dict)
                    else str(block)
                )
                for block in content
            ).strip()
        return str(content).strip()

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        if not self.settings.deepseek_api_key_value:
            raise RAGServiceError("未检测到 `DEEPSEEK_API_KEY`，请先在 `.env` 中配置。")

        os.environ["DEEPSEEK_API_KEY"] = self.settings.deepseek_api_key_value

        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError as exc:
            raise RAGServiceError("缺少 `langchain-deepseek` 依赖，请重新安装项目依赖。") from exc

        self._llm = ChatDeepSeek(
            model=self.settings.deepseek_model,
            temperature=self.settings.deepseek_temperature,
            timeout=self.settings.request_timeout_seconds,
            max_retries=self.settings.deepseek_max_retries,
        )
        return self._llm

    def _retrieve_context(self, *, question: str, top_k: int) -> list[dict[str, Any]]:
        retrieved = self.vector_store.similarity_search_with_relevance_scores(
            question,
            k=top_k,
            score_threshold=self.settings.retrieval_score_threshold,
        )
        return [
            {
                "title": doc.metadata.get("title", "未命名文档"),
                "source": doc.metadata.get("source", "unknown"),
                "score": round(float(score), 4),
                "content": doc.page_content,
                "chunk_id": doc.metadata.get("chunk_id"),
            }
            for doc, score in retrieved
        ]

    def _load_or_build_vector_store(self, *, force_rebuild: bool) -> Chroma:
        if not force_rebuild and self._is_persisted_index_current():
            vector_store = self._create_vector_store()
            persisted_count = self._count_indexed_chunks(vector_store)
            if persisted_count == len(self.chunks):
                self.index_status = "loaded"
                self.logger.info(
                    "Loaded persisted vector index",
                    extra={
                        "environment": self.settings.app_env,
                        "collection_name": self.settings.vector_store_collection_name,
                        "chunk_count": persisted_count,
                        "vector_store_dir": str(self.settings.vector_store_dir),
                    },
                )
                return vector_store

            self.logger.warning(
                "Persisted vector index count mismatch, rebuilding index",
                extra={
                    "environment": self.settings.app_env,
                    "expected_chunk_count": len(self.chunks),
                    "persisted_chunk_count": persisted_count,
                },
            )

        self.index_status = "rebuilt"
        return self._rebuild_vector_store()

    def _create_vector_store(self) -> Chroma:
        return Chroma(
            collection_name=self.settings.vector_store_collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.settings.vector_store_dir),
            collection_configuration=CHROMA_COLLECTION_CONFIGURATION,
        )

    def _rebuild_vector_store(self) -> Chroma:
        vector_store = self._create_vector_store()
        vector_store.reset_collection()
        if self.chunks:
            vector_store.add_documents(
                documents=self.chunks,
                ids=[chunk.metadata["chunk_id"] for chunk in self.chunks],
            )
        self._write_index_manifest()
        self.logger.info(
            "Rebuilt persisted vector index",
            extra={
                "environment": self.settings.app_env,
                "collection_name": self.settings.vector_store_collection_name,
                "chunk_count": len(self.chunks),
                "vector_store_dir": str(self.settings.vector_store_dir),
            },
        )
        return vector_store

    def _build_index_manifest(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for document in self.documents:
            digest.update(document.metadata.get("source", "").encode("utf-8"))
            digest.update(document.page_content.encode("utf-8"))

        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "knowledge_digest": digest.hexdigest(),
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "embedding_strategy": self.settings.embedding_strategy,
            "embedding_model_name": self.settings.embedding_model_name,
            "embedding_dimension": self.settings.embedding_dimension,
            "collection_name": self.settings.vector_store_collection_name,
        }

    def _is_persisted_index_current(self) -> bool:
        manifest_path = self.settings.vector_store_manifest_path
        if not manifest_path.exists():
            return False

        try:
            persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.logger.warning(
                "Vector index manifest is malformed, rebuilding index",
                extra={"manifest_path": str(manifest_path)},
            )
            return False

        return persisted_manifest == self.index_manifest

    def _write_index_manifest(self) -> None:
        manifest_path = self.settings.vector_store_manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(self.index_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _count_indexed_chunks(self, vector_store: Chroma) -> int:
        return len(vector_store.get(include=[])["ids"])


def sources_to_json(context_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "title": item["title"],
            "source": item["source"],
            "score": item["score"],
            "chunk_id": item.get("chunk_id"),
        }
        for item in context_docs
    ]


def _clip_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"
