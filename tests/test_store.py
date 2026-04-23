import sqlite3

import pytest

from recommender.store import Store


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "test.sqlite")


def test_init_creates_all_tables(store: Store):
    store.init_db()
    with store.connect() as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"papers", "scores", "digest_entries", "runs"} <= tables


def test_init_is_idempotent(store: Store):
    store.init_db()
    store.init_db()  # should not raise


from datetime import datetime, timezone

from recommender.models import Paper


def _paper(arxiv_id: str, **kw) -> Paper:
    base = dict(
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        abstract="abstract",
        authors=("Alice",),
        categories=("cs.LG",),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        sources=("arxiv",),
        hf_upvotes=None,
        first_seen_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )
    base.update(kw)
    return Paper(**base)


def test_upsert_inserts_new_paper(store: Store):
    store.init_db()
    new = store.upsert_papers([_paper("2604.00001")])
    assert new == 1
    with store.connect() as conn:
        row = conn.execute("SELECT * FROM papers").fetchone()
    assert row["arxiv_id"] == "2604.00001"


def test_upsert_is_idempotent_for_same_id(store: Store):
    store.init_db()
    store.upsert_papers([_paper("2604.00001")])
    new = store.upsert_papers([_paper("2604.00001")])
    assert new == 0


def test_upsert_merges_hf_signal_into_existing_row(store: Store):
    store.init_db()
    store.upsert_papers([_paper("2604.00001", sources=("arxiv",), hf_upvotes=None)])
    store.upsert_papers([_paper("2604.00001", sources=("hf",), hf_upvotes=42)])
    with store.connect() as conn:
        row = conn.execute("SELECT * FROM papers").fetchone()
    assert row["hf_upvotes"] == 42
    assert "hf" in row["sources"] and "arxiv" in row["sources"]


import json as _json

from recommender.models import Score


def test_start_run_returns_new_run_id(store: Store):
    store.init_db()
    run_id = store.start_run()
    assert isinstance(run_id, int) and run_id >= 1


def test_save_scores_persists_rows(store: Store):
    store.init_db()
    store.upsert_papers([_paper("2604.00001")])
    run_id = store.start_run()
    scores = [
        Score(
            arxiv_id="2604.00001",
            run_id=run_id,
            model="m",
            score=7.5,
            breakdown={"relevance": 7, "quality": 8, "field_importance": 7},
            why="because",
            scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        )
    ]
    store.save_scores(scores)
    with store.connect() as conn:
        row = conn.execute("SELECT * FROM scores").fetchone()
    assert row["score"] == 7.5
    saved = _json.loads(row["justification"])
    assert saved["breakdown"]["quality"] == 8
    assert saved["why"] == "because"


def test_papers_needing_scoring_excludes_already_scored_this_run(store: Store):
    store.init_db()
    store.upsert_papers([_paper("2604.00001"), _paper("2604.00002")])
    run_id = store.start_run()
    store.save_scores([
        Score(
            arxiv_id="2604.00001",
            run_id=run_id,
            model="m",
            score=5.0,
            breakdown={"relevance": 5, "quality": 5, "field_importance": 5},
            why="",
            scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        )
    ])
    needing = store.papers_needing_scoring(run_id)
    assert [p.arxiv_id for p in needing] == ["2604.00002"]


def test_record_run_finalizes_status(store: Store):
    store.init_db()
    run_id = store.start_run()
    store.record_run(run_id, status="ok", papers_seen=3, papers_scored=3, digest_date="2026-04-24")
    with store.connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["status"] == "ok"
    assert row["finished_at"] is not None
    assert row["papers_seen"] == 3
