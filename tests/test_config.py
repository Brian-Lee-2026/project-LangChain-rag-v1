from app.core.config import PROJECT_ROOT, Settings
from scripts.run_server import resolve_runtime_environment


def test_default_log_paths_are_scoped_by_environment():
    settings = Settings(app_env="prod")
    settings.ensure_directories()

    assert settings.app_env == "production"
    assert settings.app_log_path == PROJECT_ROOT / "logs" / "production" / "app.log"
    assert settings.rag_audit_path == PROJECT_ROOT / "logs" / "production" / "rag_events.jsonl"
    assert settings.feedback_log_path == PROJECT_ROOT / "logs" / "production" / "feedback.jsonl"


def test_runtime_mode_maps_to_expected_environment():
    assert resolve_runtime_environment("dev") == "development"
    assert resolve_runtime_environment("prod") == "production"
