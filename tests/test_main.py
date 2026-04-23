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
        arxiv_categories=("cs.LG",),
        arxiv_max_backfill_days=7,
        on_interest_min=1,
        on_interest_max=5,
        on_interest_threshold=7.0,
        hf_upvote_threshold_for_hot_surprise=10,
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
