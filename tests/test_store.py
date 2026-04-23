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
