from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "具身机器人知识问答系统"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = Field(default=2, ge=1, le=16)
    api_prefix: str = "/api/v1"
    cors_origins: str = "*"
    log_level: str = "INFO"

    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    vector_store_collection_name: str = "embodied_robot_knowledge"
    embedding_cache_dir: Path = PROJECT_ROOT / "data" / "model_cache"
    app_log_path: Path = PROJECT_ROOT / "logs" / "app.log"
    telemetry_db_path: Path = PROJECT_ROOT / "logs" / "telemetry.sqlite3"

    deepseek_api_key: SecretStr | None = None
    deepseek_model: str = "deepseek-chat"
    deepseek_temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    deepseek_max_retries: int = Field(default=2, ge=0, le=10)
    request_timeout_seconds: int = Field(default=60, ge=5, le=300)

    retrieval_k: int = Field(default=4, ge=1, le=12)
    retrieval_score_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    chunk_size: int = Field(default=650, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=800)
    max_history_turns: int = Field(default=6, ge=0, le=20)

    embedding_strategy: str = "local_hash"
    embedding_dimension: int = Field(default=768, ge=128, le=4096)
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_hf_local_files_only: bool = False
    embedding_fallback_to_local_hash: bool = True

    @field_validator("app_env", mode="before")
    @classmethod
    def normalize_app_env(cls, value: str) -> str:
        # 兼容 `dev` / `prod` 简写，避免环境值在不同启动方式下出现多种拼法。
        normalized = str(value).strip().lower()
        aliases = {
            "dev": "development",
            "development": "development",
            "prod": "production",
            "production": "production",
        }
        return aliases.get(normalized, normalized or "development")

    def ensure_directories(self) -> None:
        self.knowledge_base_dir = self._resolve_project_path(self.knowledge_base_dir)
        self.vector_store_dir = self._resolve_project_path(self.vector_store_dir)
        self.embedding_cache_dir = self._resolve_project_path(self.embedding_cache_dir)
        self.app_log_path = self._resolve_env_runtime_path(self.app_log_path)
        self.telemetry_db_path = self._resolve_env_runtime_path(self.telemetry_db_path)

        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self.app_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry_db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def deepseek_api_key_value(self) -> str | None:
        if self.deepseek_api_key is None:
            return None
        return self.deepseek_api_key.get_secret_value()

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def env_slug(self) -> str:
        return self.app_env.replace(" ", "_")

    def _resolve_project_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def _resolve_env_runtime_path(self, path: Path) -> Path:
        # 默认日志文件按环境分目录，避免 `development` / `production` 日志混写。
        resolved = self._resolve_project_path(path)
        log_root = (PROJECT_ROOT / "logs").resolve()
        default_log_names = {"app.log", "telemetry.sqlite3"}
        if resolved.parent == log_root and resolved.name in default_log_names:
            return log_root / self.env_slug / resolved.name
        return resolved

    @property
    def vector_store_manifest_path(self) -> Path:
        return self.vector_store_dir / "index_manifest.json"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
