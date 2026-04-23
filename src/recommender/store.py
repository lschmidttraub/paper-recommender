from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id      TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    abstract      TEXT NOT NULL,
    authors       TEXT NOT NULL,
    categories    TEXT NOT NULL,
    url           TEXT NOT NULL,
    published_at  TEXT NOT NULL,
    sources       TEXT NOT NULL,
    hf_upvotes    INTEGER,
    first_seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published_at);

CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,
    papers_seen   INTEGER,
    papers_scored INTEGER,
    digest_date   TEXT,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    arxiv_id      TEXT NOT NULL,
    run_id        INTEGER NOT NULL,
    model         TEXT NOT NULL,
    score         REAL NOT NULL,
    justification TEXT NOT NULL,
    scored_at     TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, run_id),
    FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
    FOREIGN KEY (run_id)   REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS digest_entries (
    digest_date TEXT NOT NULL,
    arxiv_id    TEXT NOT NULL,
    section     TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    PRIMARY KEY (digest_date, arxiv_id)
);
CREATE INDEX IF NOT EXISTS idx_digest_arxiv ON digest_entries(arxiv_id);
"""


class Store:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
