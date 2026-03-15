from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.services.feedback_service import FeedbackService
from app.services.rag_service import RAGService

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app_settings.ensure_directories()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 应用启动时一次性初始化日志、RAG 服务和反馈服务，避免在请求中重复创建。
        configure_logging(app_settings)
        logger = get_logger(__name__)
        app.state.settings = app_settings
        app.state.rag_service = RAGService(app_settings)
        app.state.feedback_service = FeedbackService(app_settings)
        logger.info(
            "Application started",
            extra={
                **app.state.rag_service.health().model_dump(),
                "app_log_path": str(app_settings.app_log_path),
                "rag_audit_path": str(app_settings.rag_audit_path),
                "feedback_log_path": str(app_settings.feedback_log_path),
                "environment": app_settings.app_env,
            },
        )
        yield
        logger.info("Application stopped", extra={"environment": app_settings.app_env})

    app = FastAPI(title=app_settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=app_settings.api_prefix)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def request_trace_middleware(request: Request, call_next):
        # 给每个请求补 `request_id`，便于把前端报错、应用日志和审计日志串起来。
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            get_logger("app.request").exception(
                "Unhandled request error",
                extra={"request_id": request_id, "path": str(request.url.path)},
            )
            raise
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        # 首页只作为内部测试台入口，不纳入正式接口文档。
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
