from datetime import UTC, datetime

from app.core.config import Settings
from app.core.logging import append_jsonl
from app.services.feedback_service import FeedbackService


def test_feedback_summary_aggregates_accuracy_and_rating(tmp_path):
    settings = Settings(
        feedback_log_path=tmp_path / "feedback.jsonl",
        knowledge_base_dir=tmp_path / "kb",
        vector_store_dir=tmp_path / "vector",
        rag_audit_path=tmp_path / "rag.jsonl",
        app_log_path=tmp_path / "app.log",
    )
    settings.ensure_directories()

    append_jsonl(
        settings.feedback_log_path,
        {
            "feedback_id": "1",
            "request_id": "req-1",
            "rating": 5,
            "is_accurate": True,
            "recorded_at": datetime(2026, 3, 13, tzinfo=UTC).isoformat(),
        },
    )
    append_jsonl(
        settings.feedback_log_path,
        {
            "feedback_id": "2",
            "request_id": "req-2",
            "rating": 2,
            "is_accurate": False,
            "recorded_at": datetime(2026, 3, 13, 1, tzinfo=UTC).isoformat(),
        },
    )

    summary = FeedbackService(settings).get_summary()

    assert summary.total_feedback == 2
    assert summary.accurate_votes == 1
    assert summary.inaccurate_votes == 1
    assert summary.accuracy_rate == 0.5
    assert summary.average_rating == 3.5
