import json
import logging
from datetime import UTC, datetime
from typing import Any

from concurrent_log_handler import ConcurrentRotatingFileHandler

from app.core.config import Settings

_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
    "color_message",
    "environment",
}


class JsonFormatter(logging.Formatter):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": self.settings.app_env,
            "app_name": self.settings.app_name,
        }
        # 过滤日志框架内置字段，只保留我们主动塞进来的业务上下文。
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_")
        }
        if extra:
            payload["context"] = extra
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class DevConsoleFormatter(logging.Formatter):
    def __init__(self, settings: Settings) -> None:
        super().__init__("%(asctime)s | %(levelname)s | %(name)s | %(environment)s | %(message)s")
        self.environment = settings.app_env

    def format(self, record: logging.LogRecord) -> str:
        record.environment = getattr(record, "environment", self.environment)
        return super().format(record)


def configure_logging(settings: Settings) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    # 开发环境优先可读性，生产环境控制台保持结构化格式，方便日志采集。
    if settings.is_production:
        console_handler.setFormatter(JsonFormatter(settings))
    else:
        console_handler.setFormatter(DevConsoleFormatter(settings))

    file_handler = ConcurrentRotatingFileHandler(
        settings.app_log_path,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter(settings))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
