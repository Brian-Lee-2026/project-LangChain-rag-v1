import json
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from uuid import uuid4

from app.core.config import Settings
from app.core.logging import append_jsonl, get_logger
from app.models.schemas import FeedbackRequest, FeedbackResponse, FeedbackSummary


class FeedbackService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)

    def record_feedback(self, payload: FeedbackRequest) -> FeedbackResponse:
        feedback_id = str(uuid4())
        recorded_at = datetime.now(UTC)
        append_jsonl(
            self.settings.feedback_log_path,
            {
                "environment": self.settings.app_env,
                "feedback_id": feedback_id,
                "request_id": payload.request_id,
                "conversation_id": payload.conversation_id,
                "rating": payload.rating,
                "is_accurate": payload.is_accurate,
                "notes": payload.notes,
                "recorded_at": recorded_at.isoformat(),
            },
        )
        # 反馈既写结构化日志，也保留常规应用日志，兼顾统计和排障。
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
        path = self.settings.feedback_log_path
        if not path.exists():
            return FeedbackSummary(
                total_feedback=0,
                accurate_votes=0,
                inaccurate_votes=0,
                accuracy_rate=None,
                average_rating=None,
                latest_feedback_at=None,
            )

        accurate_votes = 0
        inaccurate_votes = 0
        total_feedback = 0
        rating_sum = 0
        rating_count = 0
        latest_feedback_at: datetime | None = None

        for item in self._iter_feedback(path):
            total_feedback += 1
            if item.get("is_accurate") is True:
                accurate_votes += 1
            if item.get("is_accurate") is False:
                inaccurate_votes += 1

            # `rating` 和 `is_accurate` 分开统计，便于同时兼容星级评分与二分类准确率标注。
            rating = item.get("rating")
            if isinstance(rating, int):
                rating_sum += rating
                rating_count += 1

            recorded_at_raw = item.get("recorded_at")
            if isinstance(recorded_at_raw, str):
                recorded_at = datetime.fromisoformat(recorded_at_raw)
                if latest_feedback_at is None or recorded_at > latest_feedback_at:
                    latest_feedback_at = recorded_at

        vote_total = accurate_votes + inaccurate_votes
        accuracy_rate = round(accurate_votes / vote_total, 4) if vote_total else None
        average_rating = round(rating_sum / rating_count, 2) if rating_count else None
        return FeedbackSummary(
            total_feedback=total_feedback,
            accurate_votes=accurate_votes,
            inaccurate_votes=inaccurate_votes,
            accuracy_rate=accuracy_rate,
            average_rating=average_rating,
            latest_feedback_at=latest_feedback_at,
        )

    def _iter_feedback(self, path: Path):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except JSONDecodeError:
                    self.logger.warning(
                        "Skipped malformed feedback log line",
                        extra={"path": str(path)},
                    )
