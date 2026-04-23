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
