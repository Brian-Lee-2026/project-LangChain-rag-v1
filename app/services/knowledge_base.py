from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import PROJECT_ROOT

_SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_knowledge_documents(knowledge_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    # `title` / `source` 会在检索结果、前端引用卡片和审计日志里复用。
                    "title": _extract_title(path, text),
                    "source": _build_source_path(path, knowledge_dir),
                    "filename": path.name,
                },
            )
        )

    if not documents:
        raise FileNotFoundError(f"知识库目录 `{knowledge_dir}` 中没有可用文档。")
    return documents


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 优先按标题、段落和中文标点切分，尽量保持语义片段完整。
        separators=["\n## ", "\n### ", "\n", "。", "；", "，", " "],
    )
    chunks = splitter.split_documents(documents)
    counters: defaultdict[str, int] = defaultdict(int)
    for chunk in chunks:
        source = chunk.metadata["source"]
        counters[source] += 1
        # 为每个切片生成稳定标识，便于日志追踪和后续扩展到持久化向量库。
        chunk.metadata["chunk_id"] = f"{source}#chunk-{counters[source]}"
    return chunks


def _extract_title(path: Path, text: str) -> str:
    # 优先使用 Markdown 标题作为展示名，没有标题时再退回文件名。
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return path.stem.replace("_", " ")


def _build_source_path(path: Path, knowledge_dir: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        # 兼容测试目录或外部挂载知识库，避免路径不在项目根目录时报错。
        return str(path.relative_to(knowledge_dir))
