from app.core.config import Settings
from app.models.schemas import FeedbackRequest
from app.services.feedback_service import FeedbackService


def test_feedback_summary_aggregates_accuracy_and_rating(tmp_path):
    settings = Settings(
        knowledge_base_dir=tmp_path / "kb",
        vector_store_dir=tmp_path / "vector",
        app_log_path=tmp_path / "app.log",
        telemetry_db_path=tmp_path / "telemetry.sqlite3",
    )
    settings.ensure_directories()

    service = FeedbackService(settings)
    first = service.record_feedback(
        FeedbackRequest(
            request_id="req-001",
            rating=5,
            is_accurate=True,
        )
    )
    second = service.record_feedback(
        FeedbackRequest(
            request_id="req-002",
            rating=2,
            is_accurate=False,
        )
    )

    summary = service.get_summary()

    assert first.recorded_at <= second.recorded_at
    assert summary.total_feedback == 2
    assert summary.accurate_votes == 1
    assert summary.inaccurate_votes == 1
    assert summary.accuracy_rate == 0.5
    assert summary.average_rating == 3.5
    assert summary.latest_feedback_at is not None
