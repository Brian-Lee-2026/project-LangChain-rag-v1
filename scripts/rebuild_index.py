# ruff: noqa: E402

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.rag_service import RAGService


def main() -> None:
    settings = get_settings()
    service = RAGService(settings, force_rebuild=True)
    health = service.health()
    print("知识索引构建完成。")
    print(f"索引状态: {service.index_status}")
    print(f"文档数: {health.document_count}")
    print(f"切片数: {health.chunk_count}")
    print(f"模型: {health.model}")
    print(f"提示词版本: {health.prompt_version}")


if __name__ == "__main__":
    main()
