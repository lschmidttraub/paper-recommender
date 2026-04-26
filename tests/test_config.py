from pathlib import Path

import pytest

from recommender.config import Settings


def test_settings_from_env_applies_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("EMAIL_TO", "leo@example.com")
    monkeypatch.setenv("EMAIL_FROM", "leo@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "secret")
    # remove any cross-test pollution
    for var in ("SCORING_MODEL", "BRIDGING_MODEL", "BATCH_SIZE"):
        monkeypatch.delenv(var, raising=False)

    s = Settings.from_env(project_root=tmp_path)
    assert s.email_to == "leo@example.com"
    assert s.smtp_password == "secret"
    assert s.scoring_model == "openrouter/anthropic/claude-haiku-4-5"
    assert s.batch_size == 20
    assert s.max_tokens_per_batch == 4000
    assert s.min_scoring_success_rate == 0.5
    assert s.hot_surprise_score_max == 5.0
    assert s.bridging_score_min == 3.0
    assert s.bridging_score_max == 6.0
    assert s.db_path == tmp_path / "data" / "papers.sqlite"
    assert s.memory_md == tmp_path / "MEMORY.md"


def test_settings_respects_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("EMAIL_TO", "leo@example.com")
    monkeypatch.setenv("EMAIL_FROM", "leo@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "secret")
    monkeypatch.setenv("SCORING_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("BATCH_SIZE", "10")
    s = Settings.from_env(project_root=tmp_path)
    assert s.scoring_model == "openai/gpt-4o-mini"
    assert s.batch_size == 10


def test_settings_raises_without_required_env(monkeypatch, tmp_path):
    monkeypatch.delenv("EMAIL_TO", raising=False)
    with pytest.raises(RuntimeError):
        Settings.from_env(project_root=tmp_path)
