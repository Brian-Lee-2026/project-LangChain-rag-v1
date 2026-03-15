# ruff: noqa: E402

import argparse
import os
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

RELOAD_EXCLUDES = [
    str(PROJECT_ROOT / "data" / "model_cache"),
    str(PROJECT_ROOT / "data" / "vector_store"),
    str(PROJECT_ROOT / "logs"),
    str(PROJECT_ROOT / ".pytest_cache"),
    str(PROJECT_ROOT / ".ruff_cache"),
    *[str(path) for path in PROJECT_ROOT.glob("*.egg-info")],
]


def resolve_runtime_environment(mode: str) -> str:
    return {"dev": "development", "prod": "production"}[mode]


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 FastAPI 服务。")
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="服务启动模式。dev 开启热重载，prod 使用多进程 worker。",
    )
    args = parser.parse_args()

    # 通过启动命令明确当前环境，避免 .env 中的 APP_ENV 与运行模式打架。
    os.environ["APP_ENV"] = resolve_runtime_environment(args.mode)
    get_settings.cache_clear()
    settings = get_settings()
    is_dev = args.mode == "dev"
    configure_logging(settings)
    get_logger(__name__).info(
        "Booting server",
        extra={
            "environment": settings.app_env,
            "mode": args.mode,
            "host": settings.app_host,
            "port": settings.app_port,
            "workers": 1 if is_dev else settings.app_workers,
        },
    )

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=is_dev,
        workers=1 if is_dev else settings.app_workers,
        proxy_headers=not is_dev,
        reload_excludes=RELOAD_EXCLUDES if is_dev else None,
        log_config=None,
    )


if __name__ == "__main__":
    main()
