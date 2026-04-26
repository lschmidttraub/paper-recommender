from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from recommender.config import Settings
from recommender.main import run_pipeline
from recommender.models import Paper, Score


def _paper(arxiv_id, **kw):
    base = dict(
        arxiv_id=arxiv_id, title=f"T {arxiv_id}", abstract="abs",
        authors=("A",), categories=("cs.LG",),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        sources=("arxiv",), hf_upvotes=None,
        first_seen_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )
    base.update(kw)
    return Paper(**base)


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "MEMORY.md").write_text("# interests\n- X\n")
    return Settings(
        scoring_model="anthropic/claude-haiku-4-5",
        bridging_model="anthropic/claude-haiku-4-5",
        batch_size=20,
        max_tokens_per_batch=4000,
        min_scoring_success_rate=0.5,
        arxiv_categories=("cs.LG",),
        arxiv_max_backfill_days=7,
        on_interest_min=1,
        on_interest_max=5,
        on_interest_threshold=7.0,
        hf_upvote_threshold_for_hot_surprise=10,
        hot_surprise_score_max=5.0,
        bridging_score_min=3.0,
        bridging_score_max=6.0,
        email_to="to@example.com",
        email_from="from@example.com",
        smtp_password="pw",
        db_path=tmp_path / "data" / "t.sqlite",
        digests_dir=tmp_path / "digests",
        logs_dir=tmp_path / "logs",
        memory_md=tmp_path / "MEMORY.md",
        claude_projects_root=tmp_path / "noproj",
    )


def test_run_pipeline_happy_path(settings, mocker):
    mocker.patch("recommender.main.arxiv.fetch", return_value=[_paper("2604.00001")])
    mocker.patch("recommender.main.hf.fetch", return_value=[])
    mocker.patch("recommender.main.evaluate.score_papers", return_value=[
        Score(
            arxiv_id="2604.00001", run_id=1, model="m", score=9.0,
            breakdown={"relevance": 9, "quality": 9, "field_importance": 9},
            why="important", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        ),
    ])
    mocker.patch("recommender.main.surprise.pick_hot_outside_field", return_value=None)
    mocker.patch("recommender.main.surprise.pick_bridging", return_value=None)
    mail_mock = mocker.patch("recommender.main.mail.send")

    run_pipeline(settings, force_date="2026-04-24")

    assert (settings.digests_dir / "2026-04-24.md").exists()
    mail_mock.assert_called_once()


def test_run_pipeline_dry_run_does_not_send_email(settings, mocker):
    mocker.patch("recommender.main.arxiv.fetch", return_value=[_paper("2604.00001")])
    mocker.patch("recommender.main.hf.fetch", return_value=[])
    mocker.patch("recommender.main.evaluate.score_papers", return_value=[
        Score(
            arxiv_id="2604.00001", run_id=1, model="m", score=9.0,
            breakdown={"relevance": 9, "quality": 9, "field_importance": 9},
            why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        ),
    ])
    mocker.patch("recommender.main.surprise.pick_hot_outside_field", return_value=None)
    mocker.patch("recommender.main.surprise.pick_bridging", return_value=None)
    mail_mock = mocker.patch("recommender.main.mail.send")

    run_pipeline(settings, force_date="2026-04-24", dry_run=True)
    mail_mock.assert_not_called()
    # dry_run also does NOT persist digest_entries
    from recommender.store import Store
    store = Store(settings.db_path)
    with store.connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM digest_entries").fetchone()[0]
    assert count == 0


def test_cli_parses_force_date_and_dry_run(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("EMAIL_TO", "to@example.com")
    monkeypatch.setenv("EMAIL_FROM", "from@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "MEMORY.md").write_text("x")

    run_mock = mocker.patch("recommender.__main__.run_pipeline")
    from recommender.__main__ import main as cli_main
    cli_main(["--dry-run", "--force-date", "2026-04-24"])

    run_mock.assert_called_once()
    _args, kwargs = run_mock.call_args
    assert kwargs["dry_run"] is True
    assert kwargs["force_date"] == "2026-04-24"


def test_email_only_sends_existing_digest_without_pipeline(settings, mocker):
    date = "2026-04-24"
    settings.digests_dir.mkdir(parents=True, exist_ok=True)
    (settings.digests_dir / f"{date}.md").write_text("# Existing digest\n")

    arxiv_mock = mocker.patch("recommender.main.arxiv.fetch")
    mail_mock = mocker.patch("recommender.main.mail.send")

    run_pipeline(settings, force_date=date, email_only=True)

    arxiv_mock.assert_not_called()
    mail_mock.assert_called_once()
    _args, kwargs = mail_mock.call_args
    assert "Existing digest" in kwargs["markdown_body"]
    assert "(resend)" in kwargs["subject"]


def test_email_only_raises_if_digest_missing(settings, mocker):
    mocker.patch("recommender.main.arxiv.fetch")
    mocker.patch("recommender.main.mail.send")
    import pytest as _pytest
    with _pytest.raises(FileNotFoundError):
        run_pipeline(settings, force_date="2099-01-01", email_only=True)


def test_run_pipeline_marks_degraded_when_scoring_rate_low(settings, mocker):
    # 4 papers needing scoring; only 1 succeeds → 25% < 50% threshold
    papers = [_paper(f"2604.{i:05d}") for i in range(1, 5)]
    mocker.patch("recommender.main.arxiv.fetch", return_value=papers)
    mocker.patch("recommender.main.hf.fetch", return_value=[])
    mocker.patch("recommender.main.evaluate.score_papers", return_value=[
        Score(arxiv_id="2604.00001", run_id=1, model="m", score=9.0,
              breakdown={"relevance": 9, "quality": 9, "field_importance": 9},
              why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc)),
    ])
    mocker.patch("recommender.main.surprise.pick_hot_outside_field", return_value=None)
    mocker.patch("recommender.main.surprise.pick_bridging", return_value=None)
    mail_mock = mocker.patch("recommender.main.mail.send")

    run_pipeline(settings, force_date="2026-04-26")

    mail_mock.assert_not_called()
    from recommender.store import Store
    store = Store(settings.db_path)
    with store.connect() as conn:
        row = conn.execute("SELECT status FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    assert row["status"] == "degraded"
    # No digest entries should be marked (so papers can resurface tomorrow)
    with store.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM digest_entries").fetchone()[0]
    assert n == 0


def test_run_pipeline_remains_ok_when_no_papers_to_score(settings, mocker):
    """If there's no work to do (papers_to_score=0), do not mark as degraded."""
    mocker.patch("recommender.main.arxiv.fetch", return_value=[])
    mocker.patch("recommender.main.hf.fetch", return_value=[])
    mocker.patch("recommender.main.evaluate.score_papers", return_value=[])
    mocker.patch("recommender.main.surprise.pick_hot_outside_field", return_value=None)
    mocker.patch("recommender.main.surprise.pick_bridging", return_value=None)
    mocker.patch("recommender.main.mail.send")

    run_pipeline(settings, force_date="2026-04-26")

    from recommender.store import Store
    store = Store(settings.db_path)
    with store.connect() as conn:
        row = conn.execute("SELECT status FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    assert row["status"] == "ok"


def test_cli_parses_email_only_flag(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("EMAIL_TO", "to@example.com")
    monkeypatch.setenv("EMAIL_FROM", "from@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "MEMORY.md").write_text("x")

    run_mock = mocker.patch("recommender.__main__.run_pipeline")
    from recommender.__main__ import main as cli_main
    cli_main(["--email-only", "--force-date", "2026-04-24"])

    run_mock.assert_called_once()
    _args, kwargs = run_mock.call_args
    assert kwargs["email_only"] is True
