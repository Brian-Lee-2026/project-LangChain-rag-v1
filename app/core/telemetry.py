import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


class SQLiteTelemetryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_entries (
                    feedback_id TEXT PRIMARY KEY,
                    environment TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    conversation_id TEXT,
                    rating INTEGER,
                    is_accurate INTEGER,
                    notes TEXT,
                    recorded_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_entries_recorded_at
                    ON feedback_entries(recorded_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_entries_request_id
                    ON feedback_entries(request_id);

                CREATE TABLE IF NOT EXISTS rag_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    conversation_id TEXT,
                    question TEXT NOT NULL,
                    answer TEXT,
                    error TEXT,
                    model TEXT,
                    prompt_version TEXT,
                    latency_ms INTEGER,
                    sources_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rag_events_created_at
                    ON rag_events(created_at);
                CREATE INDEX IF NOT EXISTS idx_rag_events_request_id
                    ON rag_events(request_id);
                """
            )

    def record_feedback(
        self,
        *,
        feedback_id: str,
        environment: str,
        request_id: str,
        conversation_id: str | None,
        rating: int | None,
        is_accurate: bool | None,
        notes: str | None,
        recorded_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_entries (
                    feedback_id,
                    environment,
                    request_id,
                    conversation_id,
                    rating,
                    is_accurate,
                    notes,
                    recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    environment,
                    request_id,
                    conversation_id,
                    rating,
                    None if is_accurate is None else int(is_accurate),
                    notes,
                    recorded_at.isoformat(),
                ),
            )

    def summarize_feedback(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_feedback,
                    COALESCE(SUM(CASE WHEN is_accurate = 1 THEN 1 ELSE 0 END), 0) AS accurate_votes,
                    COALESCE(
                        SUM(CASE WHEN is_accurate = 0 THEN 1 ELSE 0 END),
                        0
                    ) AS inaccurate_votes,
                    AVG(rating) AS average_rating,
                    MAX(recorded_at) AS latest_feedback_at
                FROM feedback_entries
                """
            ).fetchone()

        if row is None:
            return {
                "total_feedback": 0,
                "accurate_votes": 0,
                "inaccurate_votes": 0,
                "accuracy_rate": None,
                "average_rating": None,
                "latest_feedback_at": None,
            }

        accurate_votes = int(row["accurate_votes"])
        inaccurate_votes = int(row["inaccurate_votes"])
        vote_total = accurate_votes + inaccurate_votes
        average_rating = float(row["average_rating"]) if row["average_rating"] is not None else None
        latest_feedback_at = (
            datetime.fromisoformat(row["latest_feedback_at"])
            if row["latest_feedback_at"]
            else None
        )
        return {
            "total_feedback": int(row["total_feedback"]),
            "accurate_votes": accurate_votes,
            "inaccurate_votes": inaccurate_votes,
            "accuracy_rate": round(accurate_votes / vote_total, 4) if vote_total else None,
            "average_rating": round(average_rating, 2) if average_rating is not None else None,
            "latest_feedback_at": latest_feedback_at,
        }

    def record_rag_event(
        self,
        *,
        event_type: str,
        environment: str,
        request_id: str,
        conversation_id: str | None,
        question: str,
        answer: str | None,
        error: str | None,
        model: str | None,
        prompt_version: str | None,
        latency_ms: int | None,
        sources: list[dict[str, Any]] | None,
        created_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rag_events (
                    event_type,
                    environment,
                    request_id,
                    conversation_id,
                    question,
                    answer,
                    error,
                    model,
                    prompt_version,
                    latency_ms,
                    sources_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    environment,
                    request_id,
                    conversation_id,
                    question,
                    answer,
                    error,
                    model,
                    prompt_version,
                    latency_ms,
                    None if sources is None else json.dumps(sources, ensure_ascii=False),
                    created_at.isoformat(),
                ),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            yield connection
            connection.commit()
        finally:
            connection.close()
