from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from recommender.models import Paper, Score

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

    def upsert_papers(self, papers: Iterable[Paper]) -> int:
        """Insert new papers; merge sources and hf_upvotes into existing rows.

        Returns the number of newly-inserted papers.
        """
        inserted = 0
        with self.connect() as conn:
            for p in papers:
                existing = conn.execute(
                    "SELECT sources, hf_upvotes FROM papers WHERE arxiv_id = ?",
                    (p.arxiv_id,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """INSERT INTO papers
                           (arxiv_id, title, abstract, authors, categories,
                            url, published_at, sources, hf_upvotes, first_seen_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            p.arxiv_id,
                            p.title,
                            p.abstract,
                            json.dumps(list(p.authors)),
                            json.dumps(list(p.categories)),
                            p.url,
                            p.published_at.isoformat(),
                            json.dumps(list(p.sources)),
                            p.hf_upvotes,
                            p.first_seen_at.isoformat(),
                        ),
                    )
                    inserted += 1
                else:
                    merged_sources = sorted(
                        set(json.loads(existing["sources"])) | set(p.sources)
                    )
                    merged_upvotes = p.hf_upvotes if p.hf_upvotes is not None else existing["hf_upvotes"]
                    conn.execute(
                        "UPDATE papers SET sources = ?, hf_upvotes = ? WHERE arxiv_id = ?",
                        (json.dumps(merged_sources), merged_upvotes, p.arxiv_id),
                    )
        return inserted

    def start_run(self) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (started_at, status) VALUES (?, 'running')",
                (datetime.now(timezone.utc).isoformat(),),
            )
            return int(cur.lastrowid)

    def record_run(
        self,
        run_id: int,
        *,
        status: str,
        papers_seen: int | None = None,
        papers_scored: int | None = None,
        digest_date: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE runs
                   SET finished_at = ?, status = ?, papers_seen = ?,
                       papers_scored = ?, digest_date = ?, error = ?
                   WHERE run_id = ?""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    status,
                    papers_seen,
                    papers_scored,
                    digest_date,
                    error,
                    run_id,
                ),
            )

    def save_scores(self, scores: Iterable[Score]) -> None:
        with self.connect() as conn:
            for s in scores:
                payload = json.dumps({"why": s.why, "breakdown": s.breakdown})
                conn.execute(
                    """INSERT OR REPLACE INTO scores
                       (arxiv_id, run_id, model, score, justification, scored_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (s.arxiv_id, s.run_id, s.model, s.score, payload, s.scored_at.isoformat()),
                )

    def papers_needing_scoring(self, run_id: int) -> list[Paper]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT p.* FROM papers p
                   WHERE NOT EXISTS (
                     SELECT 1 FROM scores s
                     WHERE s.arxiv_id = p.arxiv_id AND s.run_id = ?
                   )
                   ORDER BY p.first_seen_at DESC""",
                (run_id,),
            ).fetchall()
        return [self._row_to_paper(r) for r in rows]

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        return Paper(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            abstract=row["abstract"],
            authors=tuple(json.loads(row["authors"])),
            categories=tuple(json.loads(row["categories"])),
            url=row["url"],
            published_at=datetime.fromisoformat(row["published_at"]),
            sources=tuple(json.loads(row["sources"])),
            hf_upvotes=row["hf_upvotes"],
            first_seen_at=datetime.fromisoformat(row["first_seen_at"]),
        )
