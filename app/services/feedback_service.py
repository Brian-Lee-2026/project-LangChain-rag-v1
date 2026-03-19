from datetime import UTC, datetime
from uuid import uuid4

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.telemetry import SQLiteTelemetryStore
from app.models.schemas import FeedbackRequest, FeedbackResponse, FeedbackSummary


class FeedbackService:
    def __init__(
        self,
        settings: Settings,
        *,
        telemetry_store: SQLiteTelemetryStore | None = None,
    ) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self.telemetry_store = telemetry_store or SQLiteTelemetryStore(settings.telemetry_db_path)
        self.telemetry_store.initialize()

    def record_feedback(self, payload: FeedbackRequest) -> FeedbackResponse:
        feedback_id = str(uuid4())
        recorded_at = datetime.now(UTC)
        self.telemetry_store.record_feedback(
            feedback_id=feedback_id,
            environment=self.settings.app_env,
            request_id=payload.request_id,
            conversation_id=payload.conversation_id,
            rating=payload.rating,
            is_accurate=payload.is_accurate,
            notes=payload.notes,
            recorded_at=recorded_at,
        )
        # 反馈既写结构化日志，也保留常规应用日志，兼顾统计和排障
        self.logger.info(
            "Feedback recorded",
            extra={
                "feedback_id": feedback_id,
                "request_id": payload.request_id,
                "is_accurate": payload.is_accurate,
                "environment": self.settings.app_env,
            },
        )
        return FeedbackResponse(feedback_id=feedback_id, recorded_at=recorded_at)

    def get_summary(self) -> FeedbackSummary:
        return FeedbackSummary(**self.telemetry_store.summarize_feedback())
