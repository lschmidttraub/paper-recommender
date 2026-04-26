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

from recommender.models import DigestEntry, Score


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


def test_papers_without_scores_excludes_any_prior_score(store: Store):
    store.init_db()
    store.upsert_papers([_paper("2604.00001"), _paper("2604.00002")])
    run_id_1 = store.start_run()
    store.save_scores([
        Score(
            arxiv_id="2604.00001",
            run_id=run_id_1,
            model="m",
            score=5.0,
            breakdown={"relevance": 5, "quality": 5, "field_importance": 5},
            why="",
            scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        )
    ])
    # Even on a fresh run, 00001 should not re-appear (it was scored in run 1).
    _run_id_2 = store.start_run()
    needing = store.papers_without_scores()
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


def test_papers_for_digest_respects_threshold_and_bounds(store: Store):
    store.init_db()
    store.upsert_papers([_paper(f"2604.{i:05d}") for i in range(1, 11)])
    run_id = store.start_run()
    scores = []
    for i in range(1, 11):
        scores.append(
            Score(
                arxiv_id=f"2604.{i:05d}",
                run_id=run_id,
                model="m",
                score=float(i),  # 1..10
                breakdown={"relevance": i, "quality": i, "field_importance": i},
                why="",
                scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
        )
    store.save_scores(scores)
    picks = store.papers_for_digest(threshold=7.0, min_count=5, max_count=15)
    # papers with score >= 7 are ids 7..10 (4 papers), floor of 5 forces including id 6 as well
    assert [p.arxiv_id for p in picks] == [
        "2604.00010", "2604.00009", "2604.00008", "2604.00007", "2604.00006",
    ]


def test_papers_for_digest_excludes_already_digested(store: Store):
    store.init_db()
    store.upsert_papers([_paper(f"2604.{i:05d}") for i in range(1, 4)])
    run_id = store.start_run()
    store.save_scores([
        Score(
            arxiv_id=f"2604.{i:05d}",
            run_id=run_id, model="m", score=9.0,
            breakdown={"relevance": 9, "quality": 9, "field_importance": 9},
            why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        )
        for i in range(1, 4)
    ])
    store.mark_sent([
        DigestEntry(digest_date="2026-04-23", arxiv_id="2604.00001",
                    section="on_interest", rank=1),
    ])
    picks = store.papers_for_digest(threshold=7.0, min_count=1, max_count=10)
    ids = [p.arxiv_id for p in picks]
    assert "2604.00001" not in ids
    assert {"2604.00002", "2604.00003"} <= set(ids)


def test_mark_sent_idempotent_for_same_day(store: Store):
    store.init_db()
    store.upsert_papers([_paper("2604.00001")])
    e = DigestEntry("2026-04-24", "2604.00001", "on_interest", 1)
    store.mark_sent([e])
    store.mark_sent([e])  # INSERT OR REPLACE, must not error


def test_last_successful_finished_at_returns_none_when_no_ok_runs(store: Store):
    store.init_db()
    assert store.last_successful_finished_at() is None
    rid = store.start_run()
    store.record_run(rid, status="error")
    assert store.last_successful_finished_at() is None


def test_last_successful_finished_at_returns_latest_ok_run(store: Store):
    store.init_db()
    r1 = store.start_run()
    store.record_run(r1, status="ok")
    got = store.last_successful_finished_at()
    assert got is not None
    assert got.tzinfo is not None


def test_latest_score_and_justification_roundtrip(store: Store):
    store.init_db()
    store.upsert_papers([_paper("2604.00001")])
    rid = store.start_run()
    store.save_scores([
        Score(
            arxiv_id="2604.00001", run_id=rid, model="m", score=7.5,
            breakdown={"relevance": 7, "quality": 8, "field_importance": 7},
            why="neat", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        ),
    ])
    got = store.latest_score_and_justification("2604.00001")
    assert got is not None
    score, payload = got
    assert score == 7.5
    assert payload["breakdown"]["quality"] == 8
    assert payload["why"] == "neat"


def test_latest_score_and_justification_returns_none_for_missing(store: Store):
    store.init_db()
    assert store.latest_score_and_justification("nope") is None


def test_hot_outside_field_pick_respects_filters(store: Store):
    store.init_db()
    store.upsert_papers([
        _paper("2604.00001", sources=("arxiv", "hf"), hf_upvotes=50),
        _paper("2604.00002", sources=("arxiv",)),  # not on HF
    ])
    rid = store.start_run()
    store.save_scores([
        Score(arxiv_id="2604.00001", run_id=rid, model="m", score=3.0,
              breakdown={"relevance": 0, "quality": 0, "field_importance": 0},
              why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc)),
        Score(arxiv_id="2604.00002", run_id=rid, model="m", score=3.0,
              breakdown={"relevance": 0, "quality": 0, "field_importance": 0},
              why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc)),
    ])
    pick = store.hot_outside_field_pick(upvote_threshold=10)
    assert pick is not None and pick.arxiv_id == "2604.00001"


def test_bridging_candidates_returns_papers_in_score_band(store: Store):
    store.init_db()
    store.upsert_papers([_paper(f"2604.{i:05d}") for i in range(1, 5)])
    rid = store.start_run()
    store.save_scores([
        Score(arxiv_id="2604.00001", run_id=rid, model="m", score=2.0,
              breakdown={"relevance": 0, "quality": 0, "field_importance": 0},
              why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc)),
        Score(arxiv_id="2604.00002", run_id=rid, model="m", score=4.5,
              breakdown={"relevance": 0, "quality": 0, "field_importance": 0},
              why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc)),
        Score(arxiv_id="2604.00003", run_id=rid, model="m", score=6.0,
              breakdown={"relevance": 0, "quality": 0, "field_importance": 0},
              why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc)),
        Score(arxiv_id="2604.00004", run_id=rid, model="m", score=8.0,
              breakdown={"relevance": 0, "quality": 0, "field_importance": 0},
              why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc)),
    ])
    ids = [p.arxiv_id for p in store.bridging_candidates()]
    assert set(ids) == {"2604.00002", "2604.00003"}


def test_papers_for_digest_includes_papers_scored_in_prior_runs(store: Store):
    """Regression: papers scored on a prior run but not yet digested should
    surface in subsequent digests, not just runs where they were freshly scored."""
    store.init_db()
    store.upsert_papers([_paper("2604.00001"), _paper("2604.00002")])
    # Score on run 1
    run_id_1 = store.start_run()
    store.save_scores([
        Score(arxiv_id="2604.00001", run_id=run_id_1, model="m", score=9.0,
              breakdown={"relevance": 9, "quality": 9, "field_importance": 9},
              why="great", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc)),
        Score(arxiv_id="2604.00002", run_id=run_id_1, model="m", score=8.0,
              breakdown={"relevance": 8, "quality": 8, "field_importance": 8},
              why="good", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc)),
    ])
    # New run starts (run 2), no new scores written
    _run_id_2 = store.start_run()
    picks = store.papers_for_digest(threshold=7.0, min_count=1, max_count=10)
    assert {p.arxiv_id for p in picks} == {"2604.00001", "2604.00002"}
    assert picks[0].arxiv_id == "2604.00001"  # higher score first


def test_latest_score_and_justification_returns_most_recent_run(store: Store):
    """If a paper was scored in multiple runs, return the latest one."""
    store.init_db()
    store.upsert_papers([_paper("2604.00001")])
    r1 = store.start_run()
    store.save_scores([Score(
        arxiv_id="2604.00001", run_id=r1, model="m", score=5.0,
        breakdown={"relevance": 5, "quality": 5, "field_importance": 5},
        why="initial", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )])
    r2 = store.start_run()
    store.save_scores([Score(
        arxiv_id="2604.00001", run_id=r2, model="m", score=8.5,
        breakdown={"relevance": 9, "quality": 8, "field_importance": 9},
        why="reconsidered", scored_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )])
    got = store.latest_score_and_justification("2604.00001")
    assert got is not None
    score, payload = got
    assert score == 8.5
    assert payload["why"] == "reconsidered"
