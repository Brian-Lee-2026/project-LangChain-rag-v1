import hashlib
import os
import re
from collections.abc import Iterable

import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.logging import get_logger

_SPACE_PATTERN = re.compile(r"\s+")
logger = get_logger(__name__)


class LocalHashEmbeddings(Embeddings):
    """适用于本地演示的轻量中文友好哈希嵌入实现。"""

    def __init__(self, size: int = 768) -> None:
        self.size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.size, dtype=np.float32)
        for token in self._tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + min(len(token), 4) * 0.15
            vector[index] += sign * weight

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.astype(float).tolist()

    def _tokenize(self, text: str) -> Iterable[str]:
        normalized = _SPACE_PATTERN.sub(" ", text.strip().lower())
        if not normalized:
            return []

        char_stream = [char for char in normalized if not char.isspace()]
        # 混合词粒度、字粒度和 `n-gram`，尽量弥补轻量哈希向量对中文分词能力的不足。
        tokens: list[str] = normalized.split(" ")
        tokens.extend(char_stream)
        tokens.extend("".join(char_stream[idx : idx + 2]) for idx in range(len(char_stream) - 1))
        tokens.extend("".join(char_stream[idx : idx + 3]) for idx in range(len(char_stream) - 2))
        return [token for token in tokens if token]


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_strategy == "huggingface":
        _configure_huggingface_cache(settings)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "当前配置为 huggingface 嵌入，请先安装 `pip install -e \".[hf]\"`。"
            ) from exc
        try:
            return HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                cache_folder=str(settings.embedding_cache_dir / "sentence_transformers"),
                model_kwargs={
                    "device": settings.embedding_device,
                    "local_files_only": settings.embedding_hf_local_files_only,
                },
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            if not settings.embedding_fallback_to_local_hash:
                raise RuntimeError(
                    "HuggingFace 嵌入模型初始化失败，请检查网络、模型名或缓存目录。"
                ) from exc

            # 首次下载模型或网络抖动时，保底回退到本地向量，避免服务直接起不来。
            logger.warning(
                "HuggingFace embeddings init failed, fallback to local hash embeddings",
                extra={
                    "model_name": settings.embedding_model_name,
                    "cache_dir": str(settings.embedding_cache_dir),
                    "error": str(exc),
                },
            )
            return LocalHashEmbeddings(size=settings.embedding_dimension)

    return LocalHashEmbeddings(size=settings.embedding_dimension)


def _configure_huggingface_cache(settings: Settings) -> None:
    # 统一把模型缓存收口到项目目录，便于复用、排障和清理。
    hf_home = settings.embedding_cache_dir / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(settings.embedding_cache_dir / "transformers"))
