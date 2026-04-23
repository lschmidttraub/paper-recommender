# Paper Recommender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-user daily ML paper digest that scrapes arXiv + Hugging Face, scores papers against a hand-curated `MEMORY.md` via a multi-provider LLM layer, and delivers a ranked markdown digest by email.

**Architecture:** Modular Python pipeline run from cron. Each module has one responsibility (sources → store → interests → evaluate → surprise → render → mail). SQLite is the only persistent state. All LLM calls go through a thin LiteLLM wrapper (`llm.py`) so the scoring/bridging model is a config value. Full spec at `docs/superpowers/specs/2026-04-24-paper-recommender-design.md`.

**Tech Stack:** Python 3.11+, `uv` (deps), `litellm` (LLM abstraction), `feedparser` (arXiv ATOM), `requests` (HF), `jinja2` + `markdown` (digest rendering), `smtplib` stdlib (email), `sqlite3` stdlib, `pytest` + `pytest-mock` (tests), `python-dotenv` (config).

---

## File Structure

**New files (all under `/home/lschmidt-traub/src/paper-recommender/`):**

- `pyproject.toml`, `.env.example`, `.gitignore` (already exists), `README.md`
- `MEMORY.md` — starter template for user's interests
- `src/recommender/__init__.py` — package marker
- `src/recommender/__main__.py` — CLI entry (`python -m recommender`)
- `src/recommender/config.py` — `Settings` dataclass + env loader
- `src/recommender/models.py` — `Paper`, `Score`, `DigestEntry` dataclasses
- `src/recommender/store.py` — SQLite schema + all DB access
- `src/recommender/interests.py` — `MEMORY.md` loader + Claude Code memory scan
- `src/recommender/llm.py` — LiteLLM wrapper with `cache_control` + JSON robustness
- `src/recommender/evaluate.py` — batched scoring
- `src/recommender/surprise.py` — hot-outside-field + bridging picks
- `src/recommender/render.py` — Jinja2 digest renderer
- `src/recommender/mail.py` — SMTP sender
- `src/recommender/main.py` — pipeline orchestration
- `src/recommender/sources/__init__.py`
- `src/recommender/sources/arxiv.py` — ATOM feed fetch + parse
- `src/recommender/sources/huggingface.py` — HF Daily Papers fetch + parse
- `templates/digest.md.j2` — Jinja2 digest template
- `tests/__init__.py`, `tests/conftest.py`
- `tests/fixtures/arxiv_sample.atom`, `tests/fixtures/hf_daily_sample.json`
- `tests/test_models.py`, `tests/test_store.py`, `tests/test_arxiv.py`,
  `tests/test_huggingface.py`, `tests/test_interests.py`, `tests/test_llm.py`,
  `tests/test_evaluate.py`, `tests/test_surprise.py`, `tests/test_render.py`,
  `tests/test_mail.py`, `tests/test_config.py`, `tests/test_main.py`

**Commit cadence:** every task ends with a commit. Tests and implementation for a single unit go in the same commit.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/recommender/__init__.py`
- Create: `src/recommender/sources/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.env.example`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "paper-recommender"
version = "0.1.0"
description = "Daily ML paper digest scored against personal interests"
requires-python = ">=3.11"
dependencies = [
    "litellm>=1.50.0",
    "feedparser>=6.0.10",
    "requests>=2.31.0",
    "jinja2>=3.1.0",
    "markdown>=3.5.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-mock>=3.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/recommender"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create empty package init files**

Create `src/recommender/__init__.py`:
```python
"""Paper recommender package."""
```

Create `src/recommender/sources/__init__.py`:
```python
"""Paper sources."""
```

Create `tests/__init__.py` (empty file):
```python
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
```

- [ ] **Step 4: Create `.env.example`**

```
# LLM provider keys (set at least one; LiteLLM reads these directly)
OPENROUTER_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=

# Models (LiteLLM format)
SCORING_MODEL=openrouter/anthropic/claude-haiku-4-5
BRIDGING_MODEL=openrouter/anthropic/claude-haiku-4-5

# Email
EMAIL_TO=leo.schmidttraub@gmail.com
EMAIL_FROM=leo.schmidttraub@gmail.com
GMAIL_APP_PASSWORD=
```

- [ ] **Step 5: Run `uv sync` to verify the project installs**

Run: `uv sync --extra dev`
Expected: creates `.venv/`, installs all deps, no errors.

- [ ] **Step 6: Run pytest to verify the test infrastructure works (no tests yet)**

Run: `uv run pytest`
Expected: "no tests ran" exit 5, or "0 passed" — either is fine; we only want to confirm pytest resolves imports.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/recommender/__init__.py src/recommender/sources/__init__.py tests/__init__.py tests/conftest.py .env.example
git commit -m "scaffold paper-recommender package"
```

---

## Task 2: `models.py` — core dataclasses

**Files:**
- Create: `src/recommender/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:
```python
from datetime import datetime, timezone

from recommender.models import DigestEntry, Paper, Score


def test_paper_is_frozen():
    p = Paper(
        arxiv_id="2604.00001",
        title="A Paper",
        abstract="abstract text",
        authors=("Alice", "Bob"),
        categories=("cs.LG",),
        url="https://arxiv.org/abs/2604.00001",
        published_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        sources=("arxiv",),
        hf_upvotes=None,
        first_seen_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )
    try:
        p.title = "changed"  # type: ignore[misc]
    except Exception as e:
        assert "frozen" in str(e).lower() or "cannot assign" in str(e).lower()
    else:
        raise AssertionError("Paper should be frozen")


def test_score_has_breakdown_and_why():
    s = Score(
        arxiv_id="2604.00001",
        run_id=1,
        model="openrouter/anthropic/claude-haiku-4-5",
        score=8.5,
        breakdown={"relevance": 7, "quality": 9, "field_importance": 9},
        why="Landmark work on X",
        scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )
    assert s.score == 8.5
    assert s.breakdown["quality"] == 9


def test_digest_entry_holds_section_and_rank():
    e = DigestEntry(digest_date="2026-04-24", arxiv_id="2604.00001", section="on_interest", rank=1)
    assert e.section == "on_interest"
    assert e.rank == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recommender.models'`

- [ ] **Step 3: Implement `src/recommender/models.py`**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    url: str
    published_at: datetime
    sources: tuple[str, ...]
    hf_upvotes: int | None
    first_seen_at: datetime


@dataclass(frozen=True)
class Score:
    arxiv_id: str
    run_id: int
    model: str
    score: float
    breakdown: dict[str, int]
    why: str
    scored_at: datetime


Section = Literal["on_interest", "surprise_hot", "surprise_bridge"]


@dataclass(frozen=True)
class DigestEntry:
    digest_date: str
    arxiv_id: str
    section: Section
    rank: int
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/models.py tests/test_models.py
git commit -m "add Paper/Score/DigestEntry dataclasses"
```

---

## Task 3: `store.py` — schema + init_db + get_connection

**Files:**
- Create: `src/recommender/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:
```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/recommender/store.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/store.py tests/test_store.py
git commit -m "add Store with schema + init_db"
```

---

## Task 4: `store.py` — upsert_papers

**Files:**
- Modify: `src/recommender/store.py`
- Modify: `tests/test_store.py`

- [ ] **Step 1: Add failing tests to `tests/test_store.py`**

Append to `tests/test_store.py`:
```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v -k upsert`
Expected: FAIL — `Store has no attribute 'upsert_papers'`.

- [ ] **Step 3: Implement `upsert_papers` in `src/recommender/store.py`**

Add at the top of the file:
```python
import json
from collections.abc import Iterable
```

Add method to the `Store` class:
```python
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
```

Also add `from recommender.models import Paper` to the imports at the top.

- [ ] **Step 4: Run the upsert tests**

Run: `uv run pytest tests/test_store.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/store.py tests/test_store.py
git commit -m "add Store.upsert_papers with source merge"
```

---

## Task 5: `store.py` — runs lifecycle + save_scores + papers_needing_scoring

**Files:**
- Modify: `src/recommender/store.py`
- Modify: `tests/test_store.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_store.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_store.py -v -k "run or save_scores or needing"`
Expected: FAIL — missing methods.

- [ ] **Step 3: Implement in `src/recommender/store.py`**

Add at the top:
```python
from datetime import datetime, timezone

from recommender.models import Score
```

Add methods to `Store`:
```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_store.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/store.py tests/test_store.py
git commit -m "add Store run lifecycle, save_scores, papers_needing_scoring"
```

---

## Task 6: `store.py` — digest pickers (top_scoring_today, mark_sent, helpers)

**Files:**
- Modify: `src/recommender/store.py`
- Modify: `tests/test_store.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_store.py`:
```python
def test_top_scoring_respects_threshold_and_bounds(store: Store):
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
    picks = store.top_scoring_today(run_id, threshold=7.0, min_count=5, max_count=15)
    # papers with score >= 7 are ids 7..10 (4 papers), floor of 5 forces including id 6 as well
    assert [p.arxiv_id for p in picks] == [
        "2604.00010", "2604.00009", "2604.00008", "2604.00007", "2604.00006",
    ]


def test_top_scoring_excludes_already_digested(store: Store):
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
    picks = store.top_scoring_today(run_id, threshold=7.0, min_count=1, max_count=10)
    ids = [p.arxiv_id for p in picks]
    assert "2604.00001" not in ids
    assert {"2604.00002", "2604.00003"} <= set(ids)


def test_mark_sent_idempotent_for_same_day(store: Store):
    store.init_db()
    store.upsert_papers([_paper("2604.00001")])
    e = DigestEntry("2026-04-24", "2604.00001", "on_interest", 1)
    store.mark_sent([e])
    store.mark_sent([e])  # INSERT OR REPLACE, must not error
```

Also add to the existing imports at the top of the test file:
```python
from recommender.models import DigestEntry
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_store.py -v -k "top_scoring or mark_sent"`
Expected: FAIL — missing methods.

- [ ] **Step 3: Implement in `src/recommender/store.py`**

Add methods to `Store`:
```python
    def top_scoring_today(
        self,
        run_id: int,
        *,
        threshold: float,
        min_count: int,
        max_count: int,
    ) -> list[Paper]:
        """Return up to max_count papers for the digest, ordered by score desc.

        Includes papers with score >= threshold. If fewer than min_count qualify,
        fills up to min_count with the next-highest scoring papers. Always
        excludes papers that already appear in any past digest_entries row.
        """
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT p.*, s.score
                   FROM papers p
                   JOIN scores s ON s.arxiv_id = p.arxiv_id
                   WHERE s.run_id = ?
                     AND NOT EXISTS (
                       SELECT 1 FROM digest_entries d WHERE d.arxiv_id = p.arxiv_id
                     )
                   ORDER BY s.score DESC""",
                (run_id,),
            ).fetchall()

        ranked = [(r, r["score"]) for r in rows]
        above = [r for r, sc in ranked if sc >= threshold]
        if len(above) >= min_count:
            picked = above[:max_count]
        else:
            picked = [r for r, _ in ranked[:min_count]]
        return [self._row_to_paper(r) for r in picked]

    def mark_sent(self, entries: Iterable[DigestEntry]) -> None:
        with self.connect() as conn:
            for e in entries:
                conn.execute(
                    """INSERT OR REPLACE INTO digest_entries
                       (digest_date, arxiv_id, section, rank)
                       VALUES (?, ?, ?, ?)""",
                    (e.digest_date, e.arxiv_id, e.section, e.rank),
                )
```

Add `from recommender.models import DigestEntry, Paper, Score` to imports.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_store.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/store.py tests/test_store.py
git commit -m "add Store digest pickers: top_scoring_today, mark_sent"
```

---

## Task 7: `sources/arxiv.py` — fetch + parse

**Files:**
- Create: `src/recommender/sources/arxiv.py`
- Create: `tests/test_arxiv.py`
- Create: `tests/fixtures/arxiv_sample.atom`

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/arxiv_sample.atom` (minimal ATOM feed with two entries):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2604.00001v1</id>
    <title>Sparse Autoencoders Find Interpretable Features</title>
    <summary>We train sparse autoencoders on transformer activations.</summary>
    <published>2026-04-24T00:00:00Z</published>
    <author><name>Alice Author</name></author>
    <author><name>Bob Author</name></author>
    <link rel="alternate" href="http://arxiv.org/abs/2604.00001v1"/>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2604.00002v2</id>
    <title>Scaling Laws Revisited</title>
    <summary>We revisit scaling laws under new conditions.</summary>
    <published>2026-04-24T00:00:00Z</published>
    <author><name>Carol Author</name></author>
    <link rel="alternate" href="http://arxiv.org/abs/2604.00002v2"/>
    <category term="stat.ML"/>
  </entry>
</feed>
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_arxiv.py`:
```python
from datetime import datetime, timezone

from recommender.sources.arxiv import build_query, parse_atom


def test_parse_atom_normalizes_papers(fixtures_dir):
    feed_text = (fixtures_dir / "arxiv_sample.atom").read_text()
    papers = parse_atom(feed_text, now=datetime(2026, 4, 24, tzinfo=timezone.utc))
    assert len(papers) == 2
    p0 = papers[0]
    assert p0.arxiv_id == "2604.00001"   # version suffix stripped
    assert p0.title == "Sparse Autoencoders Find Interpretable Features"
    assert p0.authors == ("Alice Author", "Bob Author")
    assert "cs.LG" in p0.categories and "cs.AI" in p0.categories
    assert p0.sources == ("arxiv",)
    assert p0.hf_upvotes is None


def test_build_query_includes_all_categories_and_date_window():
    q = build_query(
        categories=("cs.LG", "cs.AI"),
        since=datetime(2026, 4, 20, tzinfo=timezone.utc),
        until=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )
    assert "cs.LG" in q and "cs.AI" in q
    assert "202604200000" in q.replace("-", "").replace(":", "").replace(" ", "")
    assert "202604240000" in q.replace("-", "").replace(":", "").replace(" ", "")
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_arxiv.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement `src/recommender/sources/arxiv.py`**

```python
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import feedparser
import requests

from recommender.models import Paper

log = logging.getLogger(__name__)

_ARXIV_API = "http://export.arxiv.org/api/query"
_VERSION_SUFFIX = re.compile(r"v\d+$")
_ID_FROM_URL = re.compile(r"arxiv\.org/abs/([^/]+?)(?:v\d+)?$")


def build_query(
    categories: tuple[str, ...],
    since: datetime,
    until: datetime,
    max_results: int = 2000,
) -> str:
    cat_clause = "+OR+".join(f"cat:{c}" for c in categories)
    date_fmt = "%Y%m%d%H%M"
    window = f"submittedDate:[{since.strftime(date_fmt)}+TO+{until.strftime(date_fmt)}]"
    search = f"({cat_clause})+AND+{window}"
    return (
        f"{_ARXIV_API}?search_query={search}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )


def fetch(
    categories: tuple[str, ...],
    since: datetime,
    until: datetime | None = None,
    *,
    session: requests.Session | None = None,
) -> list[Paper]:
    until = until or datetime.now(timezone.utc)
    url = build_query(categories, since, until)
    sess = session or requests.Session()
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    return parse_atom(resp.text, now=datetime.now(timezone.utc))


def parse_atom(feed_text: str, *, now: datetime) -> list[Paper]:
    parsed = feedparser.parse(feed_text)
    papers: list[Paper] = []
    for entry in parsed.entries:
        arxiv_id = _extract_id(entry)
        if not arxiv_id:
            log.warning("Skipping arxiv entry without id: %r", entry.get("title"))
            continue
        try:
            published = datetime.fromisoformat(entry.published.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            published = now
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_clean(entry.get("title", "")),
                abstract=_clean(entry.get("summary", "")),
                authors=tuple(a.get("name", "") for a in entry.get("authors", [])),
                categories=tuple(
                    t.get("term", "") for t in entry.get("tags", []) if t.get("term")
                ),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                published_at=published,
                sources=("arxiv",),
                hf_upvotes=None,
                first_seen_at=now,
            )
        )
    return papers


def _extract_id(entry) -> str | None:
    raw = entry.get("id", "")
    m = _ID_FROM_URL.search(raw)
    if not m:
        return None
    return _VERSION_SUFFIX.sub("", m.group(1))


def _clean(s: str) -> str:
    return " ".join(s.split())
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_arxiv.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/recommender/sources/arxiv.py tests/test_arxiv.py tests/fixtures/arxiv_sample.atom
git commit -m "add arXiv source: fetch + ATOM parser"
```

---

## Task 8: `sources/huggingface.py` — fetch + parse

**Files:**
- Create: `src/recommender/sources/huggingface.py`
- Create: `tests/test_huggingface.py`
- Create: `tests/fixtures/hf_daily_sample.json`

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/hf_daily_sample.json`:
```json
[
  {
    "paper": {
      "id": "2604.00001",
      "title": "Sparse Autoencoders Find Interpretable Features",
      "summary": "We train SAEs on transformer activations.",
      "authors": [{"name": "Alice Author"}, {"name": "Bob Author"}],
      "publishedAt": "2026-04-24T00:00:00.000Z",
      "upvotes": 42
    }
  },
  {
    "paper": {
      "id": "2604.00005",
      "title": "No arxiv link paper",
      "summary": "Summary.",
      "authors": [{"name": "Solo"}],
      "publishedAt": "2026-04-24T00:00:00.000Z",
      "upvotes": 3
    }
  }
]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_huggingface.py`:
```python
import json
from datetime import datetime, timezone

from recommender.sources.huggingface import parse_daily


def test_parse_daily_normalizes_papers_with_upvotes(fixtures_dir):
    raw = json.loads((fixtures_dir / "hf_daily_sample.json").read_text())
    papers = parse_daily(raw, now=datetime(2026, 4, 24, tzinfo=timezone.utc))
    assert len(papers) == 2
    p0 = papers[0]
    assert p0.arxiv_id == "2604.00001"
    assert p0.hf_upvotes == 42
    assert p0.sources == ("hf",)
    assert p0.authors == ("Alice Author", "Bob Author")
    assert p0.url == "https://arxiv.org/abs/2604.00001"
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_huggingface.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement `src/recommender/sources/huggingface.py`**

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from recommender.models import Paper

log = logging.getLogger(__name__)

_HF_DAILY_API = "https://huggingface.co/api/daily_papers"


def fetch(session: requests.Session | None = None) -> list[Paper]:
    sess = session or requests.Session()
    resp = sess.get(_HF_DAILY_API, timeout=30)
    resp.raise_for_status()
    return parse_daily(resp.json(), now=datetime.now(timezone.utc))


def parse_daily(raw: list[dict], *, now: datetime) -> list[Paper]:
    papers: list[Paper] = []
    for entry in raw:
        paper = entry.get("paper") or entry
        arxiv_id = paper.get("id")
        if not arxiv_id:
            continue
        try:
            published = datetime.fromisoformat(paper["publishedAt"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            published = now
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=(paper.get("title") or "").strip(),
                abstract=(paper.get("summary") or "").strip(),
                authors=tuple(
                    a.get("name", "") for a in paper.get("authors", []) if a.get("name")
                ),
                categories=(),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                published_at=published,
                sources=("hf",),
                hf_upvotes=paper.get("upvotes"),
                first_seen_at=now,
            )
        )
    return papers
```

- [ ] **Step 5: Run the test**

Run: `uv run pytest tests/test_huggingface.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/recommender/sources/huggingface.py tests/test_huggingface.py tests/fixtures/hf_daily_sample.json
git commit -m "add Hugging Face Daily Papers source"
```

---

## Task 9: `interests.py` — MEMORY.md + Claude Code memory scan

**Files:**
- Create: `src/recommender/interests.py`
- Create: `tests/test_interests.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_interests.py`:
```python
from pathlib import Path

from recommender.interests import load


def test_load_reads_memory_md_verbatim(tmp_path: Path):
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text("# interests\nfoo\nbar\n")
    primary, secondary = load(memory_md=memory_md, claude_projects_root=tmp_path / "nope")
    assert primary == "# interests\nfoo\nbar\n"
    assert secondary == ""


def test_load_scans_claude_code_memory_dirs(tmp_path: Path):
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text("primary")

    projects = tmp_path / "projects"
    proj_a = projects / "-home-user-src-repo-a" / "memory"
    proj_a.mkdir(parents=True)
    (proj_a / "MEMORY.md").write_text("a-memory")
    (proj_a / "project_foo.md").write_text("proj-foo-note")

    proj_b = projects / "-home-user-src-repo-b" / "memory"
    proj_b.mkdir(parents=True)
    (proj_b / "user_role.md").write_text("user-role-b")

    (projects / "not-a-project-dir").mkdir()   # no memory/, should be skipped

    primary, secondary = load(memory_md=memory_md, claude_projects_root=projects)
    assert primary == "primary"
    assert "repo-a" in secondary
    assert "a-memory" in secondary
    assert "proj-foo-note" in secondary
    assert "repo-b" in secondary
    assert "user-role-b" in secondary
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_interests.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/recommender/interests.py`**

```python
from __future__ import annotations

from pathlib import Path

_RELEVANT_PREFIXES = ("MEMORY.md", "project_", "user_", "feedback_", "reference_")


def load(memory_md: Path, claude_projects_root: Path) -> tuple[str, str]:
    primary = memory_md.read_text() if memory_md.exists() else ""
    secondary = _scan_claude_memory(claude_projects_root)
    return primary, secondary


def _scan_claude_memory(root: Path) -> str:
    if not root.exists():
        return ""
    blocks: list[str] = []
    for project_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        memory_dir = project_dir / "memory"
        if not memory_dir.is_dir():
            continue
        project_name = project_dir.name
        files = sorted(
            f for f in memory_dir.iterdir()
            if f.is_file() and f.suffix == ".md"
            and (f.name == "MEMORY.md" or any(f.name.startswith(p) for p in _RELEVANT_PREFIXES))
        )
        if not files:
            continue
        body = "\n\n".join(f"## {f.name}\n{f.read_text().strip()}" for f in files)
        blocks.append(f"# Project: {project_name}\n\n{body}")
    return "\n\n---\n\n".join(blocks)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_interests.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/interests.py tests/test_interests.py
git commit -m "add interests loader with Claude Code memory scan"
```

---

## Task 10: `llm.py` — LiteLLM wrapper with cache_control + JSON robustness

**Files:**
- Create: `src/recommender/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm.py`:
```python
import json
from unittest.mock import MagicMock

import pytest

from recommender.llm import complete_json_array, complete_json_object


@pytest.fixture
def mock_litellm(mocker):
    m = mocker.patch("recommender.llm.litellm_completion")
    # default: return a valid JSON array
    m.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='[{"id": "1", "score": 7}]'))])
    return m


def test_complete_json_array_parses_strict(mock_litellm):
    out = complete_json_array(model="anthropic/claude-haiku-4-5", messages=[{"role": "user", "content": "x"}])
    assert out == [{"id": "1", "score": 7}]


def test_complete_json_array_extracts_array_from_noisy_output(mock_litellm):
    mock_litellm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='Here you go: [{"id": "1"}]  thanks!'))]
    )
    out = complete_json_array(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
    assert out == [{"id": "1"}]


def test_complete_json_array_retries_on_parse_failure(mock_litellm):
    mock_litellm.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="not json"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content='[{"id": "2"}]'))]),
    ]
    out = complete_json_array(model="anthropic/claude-haiku-4-5", messages=[{"role": "user", "content": "x"}])
    assert out == [{"id": "2"}]
    assert mock_litellm.call_count == 2


def test_complete_json_array_raises_after_final_failure(mock_litellm):
    mock_litellm.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="still not json"))])
    with pytest.raises(ValueError):
        complete_json_array(model="anthropic/claude-haiku-4-5", messages=[{"role": "user", "content": "x"}])


def test_cache_control_applied_for_anthropic_models(mock_litellm):
    cacheable = "LONG CACHEABLE BLOCK"
    complete_json_array(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "x"}],
        cacheable_prefix=cacheable,
    )
    args, kwargs = mock_litellm.call_args
    msgs = kwargs.get("messages") or args[1]
    # First user message should include the cacheable block with cache_control
    parts = msgs[0]["content"]
    assert isinstance(parts, list)
    has_cache_marker = any(
        p.get("cache_control", {}).get("type") == "ephemeral"
        for p in parts if isinstance(p, dict)
    )
    assert has_cache_marker
    assert any(cacheable in (p.get("text") or "") for p in parts if isinstance(p, dict))


def test_cache_control_not_applied_for_openai(mock_litellm):
    complete_json_array(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
        cacheable_prefix="LONG",
    )
    args, kwargs = mock_litellm.call_args
    msgs = kwargs.get("messages") or args[1]
    # For non-anthropic, we pass a plain string prefix (no cache_control marker)
    content = msgs[0]["content"]
    assert isinstance(content, str)
    assert "LONG" in content


def test_complete_json_object_parses_object(mock_litellm):
    mock_litellm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"id": "1", "bridge_reason": "because"}'))]
    )
    out = complete_json_object(model="anthropic/claude-haiku-4-5", messages=[{"role": "user", "content": "x"}])
    assert out == {"id": "1", "bridge_reason": "because"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/recommender/llm.py`**

```python
from __future__ import annotations

import json
import logging
import re
from typing import Any

import litellm
from litellm import completion as litellm_completion  # re-exported so tests can monkeypatch

log = logging.getLogger(__name__)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("anthropic/") or model.startswith("openrouter/anthropic/")


def _prepend_cacheable(messages: list[dict], cacheable_prefix: str, model: str) -> list[dict]:
    if not cacheable_prefix:
        return messages
    if _is_anthropic_model(model):
        # Anthropic content-block form with cache_control marker
        first_user_idx = next(i for i, m in enumerate(messages) if m["role"] == "user")
        original = messages[first_user_idx]["content"]
        original_text = original if isinstance(original, str) else original[0].get("text", "")
        new_content = [
            {
                "type": "text",
                "text": cacheable_prefix,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": original_text},
        ]
        out = list(messages)
        out[first_user_idx] = {"role": "user", "content": new_content}
        return out
    # Non-anthropic: plain string concatenation, relying on automatic prefix caching.
    first_user_idx = next(i for i, m in enumerate(messages) if m["role"] == "user")
    original = messages[first_user_idx]["content"]
    if not isinstance(original, str):
        original = "".join(p.get("text", "") for p in original if isinstance(p, dict))
    out = list(messages)
    out[first_user_idx] = {"role": "user", "content": f"{cacheable_prefix}\n\n{original}"}
    return out


def _call_llm(model: str, messages: list[dict], **kwargs) -> str:
    resp = litellm_completion(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content or ""


def _extract_json(raw: str, pattern: re.Pattern[str]) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = pattern.search(raw)
    if not m:
        raise ValueError(f"No JSON match in LLM output: {raw[:200]!r}")
    return json.loads(m.group(0))


def complete_json_array(
    *,
    model: str,
    messages: list[dict],
    cacheable_prefix: str = "",
    max_retries: int = 1,
    **kwargs,
) -> list[dict]:
    msgs = _prepend_cacheable(messages, cacheable_prefix, model)
    last_err: Exception | None = None
    for _ in range(max_retries + 1):
        raw = _call_llm(model, msgs, **kwargs)
        try:
            parsed = _extract_json(raw, _JSON_ARRAY_RE)
            if isinstance(parsed, list):
                return parsed
            raise ValueError("Expected JSON array, got object")
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            log.warning("LLM JSON array parse failed: %s", e)
    raise ValueError(f"LLM never returned parseable JSON array: {last_err}")


def complete_json_object(
    *,
    model: str,
    messages: list[dict],
    cacheable_prefix: str = "",
    max_retries: int = 1,
    **kwargs,
) -> dict:
    msgs = _prepend_cacheable(messages, cacheable_prefix, model)
    last_err: Exception | None = None
    for _ in range(max_retries + 1):
        raw = _call_llm(model, msgs, **kwargs)
        try:
            parsed = _extract_json(raw, _JSON_OBJECT_RE)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Expected JSON object, got array")
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            log.warning("LLM JSON object parse failed: %s", e)
    raise ValueError(f"LLM never returned parseable JSON object: {last_err}")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_llm.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/llm.py tests/test_llm.py
git commit -m "add LiteLLM wrapper with cache_control + JSON parsing"
```

---

## Task 11: `evaluate.py` — batched scoring with rubric

**Files:**
- Create: `src/recommender/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:
```python
from datetime import datetime, timezone

import pytest

from recommender.evaluate import SCORING_RUBRIC, build_cacheable_prefix, build_user_message, score_papers
from recommender.models import Paper


def _paper(arxiv_id: str, hf_upvotes: int | None = None) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        abstract="Abstract text.",
        authors=("Alice",),
        categories=("cs.LG",),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        sources=("arxiv",),
        hf_upvotes=hf_upvotes,
        first_seen_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


def test_rubric_mentions_three_factors_and_bands():
    assert "RELEVANCE" in SCORING_RUBRIC
    assert "QUALITY" in SCORING_RUBRIC
    assert "FIELD IMPORTANCE" in SCORING_RUBRIC
    assert "0-2" in SCORING_RUBRIC and "3-5" in SCORING_RUBRIC
    assert "10" in SCORING_RUBRIC


def test_build_cacheable_prefix_contains_interests_and_rubric():
    prefix = build_cacheable_prefix(primary="MY INTERESTS", secondary="SECONDARY")
    assert "<user_interests>" in prefix
    assert "MY INTERESTS" in prefix
    assert "<secondary_signals>" in prefix
    assert "SECONDARY" in prefix
    assert "<scoring_rubric>" in prefix


def test_build_user_message_includes_all_papers_as_json():
    msg = build_user_message([_paper("2604.00001", hf_upvotes=5), _paper("2604.00002")])
    assert "2604.00001" in msg and "2604.00002" in msg
    assert '"hf_upvotes": 5' in msg


def test_score_papers_batches_of_batch_size(mocker):
    mock = mocker.patch("recommender.evaluate.complete_json_array")
    mock.side_effect = [
        [{"id": "2604.00001", "score": 7.0,
          "breakdown": {"relevance": 7, "quality": 7, "field_importance": 7},
          "why": "reason"}],
        [{"id": "2604.00002", "score": 3.0,
          "breakdown": {"relevance": 3, "quality": 3, "field_importance": 3},
          "why": "reason"}],
    ]
    papers = [_paper("2604.00001"), _paper("2604.00002")]
    scores = score_papers(
        papers,
        primary="P",
        secondary="S",
        run_id=1,
        model="openrouter/anthropic/claude-haiku-4-5",
        batch_size=1,  # force two calls
    )
    assert mock.call_count == 2
    assert {s.arxiv_id for s in scores} == {"2604.00001", "2604.00002"}
    s1 = next(s for s in scores if s.arxiv_id == "2604.00001")
    assert s1.score == 7.0
    assert s1.breakdown["quality"] == 7
    assert s1.why == "reason"


def test_score_papers_skips_papers_whose_batch_fails(mocker):
    mock = mocker.patch("recommender.evaluate.complete_json_array")
    mock.side_effect = [ValueError("parse fail"), [
        {"id": "2604.00002", "score": 3.0,
         "breakdown": {"relevance": 3, "quality": 3, "field_importance": 3},
         "why": "r"},
    ]]
    papers = [_paper("2604.00001"), _paper("2604.00002")]
    scores = score_papers(
        papers, primary="P", secondary="", run_id=1,
        model="anthropic/claude-haiku-4-5", batch_size=1,
    )
    assert {s.arxiv_id for s in scores} == {"2604.00002"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/recommender/evaluate.py`**

```python
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from recommender.llm import complete_json_array
from recommender.models import Paper, Score

log = logging.getLogger(__name__)

SCORING_RUBRIC = """<scoring_rubric>
Score each paper 0-10 as a JOINT assessment of three factors:
  (a) RELEVANCE        — match to <user_interests>
  (b) QUALITY          — novelty, methodological rigor, specificity of claims,
                         clarity of contribution (inferred from abstract)
  (c) FIELD IMPORTANCE — potential to be a landmark/foundational paper even
                         in a field the user did not list. Signals include
                         bold novel claims, new paradigms, author track
                         record, and community attention (hf_upvotes).

Band guide:
  0-2  Unimportant: weak AND off-topic / low general merit
  3-5  Tangential: off-topic with some merit, OR on-topic but mediocre
  6-7  Solidly relevant + reasonable quality, OR important in nearby field
  8-9  Strongly relevant + good quality, OR a landmark-level paper in any field
  10   Exceptional: generational in user's field, OR field-wide landmark
       that connects to the user's interests

Apply the "Not interested in" section of <user_interests> as a hard signal —
deduct 3+ points, UNLESS the paper is genuinely landmark-level (>=8), in
which case field importance wins.

Respond with a JSON ARRAY, one object per paper, in the same order as the
input. Each object MUST have: id, score (0-10 float), breakdown
({relevance, quality, field_importance} ints 0-10), why (one sentence).
</scoring_rubric>"""


_SYSTEM_PROMPT = (
    "You are a research assistant ranking ML papers for a specific user. "
    "Return ONLY a valid JSON array — no prose, no markdown fences."
)


def build_cacheable_prefix(primary: str, secondary: str) -> str:
    return (
        f"<user_interests>\n{primary}\n</user_interests>\n\n"
        f"<secondary_signals>\n{secondary}\n</secondary_signals>\n\n"
        f"{SCORING_RUBRIC}"
    )


def build_user_message(batch: list[Paper]) -> str:
    items = [
        {
            "id": p.arxiv_id,
            "title": p.title,
            "abstract": p.abstract,
            "categories": list(p.categories),
            "hf_upvotes": p.hf_upvotes,
        }
        for p in batch
    ]
    return f"<papers>\n{json.dumps(items, indent=2)}\n</papers>"


def score_papers(
    papers: list[Paper],
    *,
    primary: str,
    secondary: str,
    run_id: int,
    model: str,
    batch_size: int = 20,
) -> list[Score]:
    prefix = build_cacheable_prefix(primary, secondary)
    results: list[Score] = []
    now = datetime.now(timezone.utc)
    for start in range(0, len(papers), batch_size):
        batch = papers[start : start + batch_size]
        user_msg = build_user_message(batch)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            raw = complete_json_array(
                model=model, messages=messages, cacheable_prefix=prefix, max_retries=1
            )
        except Exception as e:
            log.warning("Scoring batch %d-%d failed: %s", start, start + len(batch), e)
            continue
        by_id = {p.arxiv_id: p for p in batch}
        for item in raw:
            arxiv_id = item.get("id")
            if arxiv_id not in by_id:
                log.warning("Score for unknown id %r ignored", arxiv_id)
                continue
            try:
                results.append(
                    Score(
                        arxiv_id=arxiv_id,
                        run_id=run_id,
                        model=model,
                        score=float(item["score"]),
                        breakdown={
                            k: int(item["breakdown"].get(k, 0))
                            for k in ("relevance", "quality", "field_importance")
                        },
                        why=str(item.get("why", ""))[:500],
                        scored_at=now,
                    )
                )
            except (KeyError, ValueError, TypeError) as e:
                log.warning("Malformed score for %s: %s", arxiv_id, e)
    return results
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/evaluate.py tests/test_evaluate.py
git commit -m "add batched scoring with rubric + cacheable prefix"
```

---

## Task 12: `surprise.py` — hot-outside-field (SQL) + bridging (LLM)

**Files:**
- Create: `src/recommender/surprise.py`
- Create: `tests/test_surprise.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_surprise.py`:
```python
from datetime import datetime, timezone

import pytest

from recommender.models import Paper, Score
from recommender.store import Store
from recommender.surprise import pick_bridging, pick_hot_outside_field


def _paper(arxiv_id, *, sources=("arxiv",), hf_upvotes=None):
    return Paper(
        arxiv_id=arxiv_id,
        title=f"T {arxiv_id}",
        abstract="abs",
        authors=("A",),
        categories=("cs.LG",),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        sources=sources,
        hf_upvotes=hf_upvotes,
        first_seen_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


def _score(arxiv_id, run_id, value):
    return Score(
        arxiv_id=arxiv_id, run_id=run_id, model="m", score=value,
        breakdown={"relevance": 0, "quality": 0, "field_importance": 0},
        why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


@pytest.fixture
def populated_store(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    store.init_db()
    store.upsert_papers([
        _paper("2604.00001", sources=("arxiv", "hf"), hf_upvotes=50),
        _paper("2604.00002", sources=("arxiv", "hf"), hf_upvotes=25),
        _paper("2604.00003", sources=("arxiv",)),  # not on HF
        _paper("2604.00004", sources=("arxiv", "hf"), hf_upvotes=5),  # under threshold
    ])
    run_id = store.start_run()
    store.save_scores([
        _score("2604.00001", run_id, 3.0),  # low score, hot HF => eligible
        _score("2604.00002", run_id, 4.5),  # lower upvotes but eligible
        _score("2604.00003", run_id, 2.0),  # not on HF => ineligible
        _score("2604.00004", run_id, 2.0),  # HF upvotes under threshold => ineligible
    ])
    return store, run_id


def test_pick_hot_outside_field_returns_top_upvoted_low_score(populated_store):
    store, run_id = populated_store
    pick = pick_hot_outside_field(store, run_id=run_id, upvote_threshold=10)
    assert pick is not None
    assert pick.arxiv_id == "2604.00001"


def test_pick_hot_outside_field_returns_none_when_no_candidates(populated_store):
    store, run_id = populated_store
    pick = pick_hot_outside_field(store, run_id=run_id, upvote_threshold=1000)
    assert pick is None


def test_pick_bridging_returns_none_when_too_few_candidates(populated_store, mocker):
    store, run_id = populated_store
    mock = mocker.patch("recommender.surprise.complete_json_object")
    # Only 2 papers in 3-6 band (0001 at 3.0, 0002 at 4.5); below min_candidates default 10.
    pick = pick_bridging(
        store, run_id=run_id, primary="interests",
        model="anthropic/claude-haiku-4-5", min_candidates=10,
    )
    assert pick is None
    mock.assert_not_called()


def test_pick_bridging_calls_llm_and_returns_chosen(populated_store, mocker):
    store, run_id = populated_store
    # Add more mid-band papers to clear the min_candidates bar
    extra = [_paper(f"2604.{i:05d}") for i in range(10, 30)]
    store.upsert_papers(extra)
    store.save_scores([_score(p.arxiv_id, run_id, 4.0) for p in extra])

    mock = mocker.patch("recommender.surprise.complete_json_object")
    mock.return_value = {"id": "2604.00010", "bridge_reason": "methodological parallel"}
    pick = pick_bridging(
        store, run_id=run_id, primary="interests",
        model="anthropic/claude-haiku-4-5", min_candidates=10,
    )
    assert pick is not None
    paper, reason = pick
    assert paper.arxiv_id == "2604.00010"
    assert reason == "methodological parallel"
    mock.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_surprise.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/recommender/surprise.py`**

```python
from __future__ import annotations

import json
import logging
import random

from recommender.llm import complete_json_object
from recommender.models import Paper
from recommender.store import Store

log = logging.getLogger(__name__)


def pick_hot_outside_field(
    store: Store,
    *,
    run_id: int,
    upvote_threshold: int,
) -> Paper | None:
    """Highest-upvoted HF paper from this run that scored <5 and is not yet digested."""
    with store.connect() as conn:
        row = conn.execute(
            """SELECT p.*
               FROM papers p
               JOIN scores s ON s.arxiv_id = p.arxiv_id
               WHERE s.run_id = ?
                 AND p.hf_upvotes IS NOT NULL
                 AND p.hf_upvotes >= ?
                 AND s.score < 5
                 AND EXISTS (SELECT 1 FROM json_each(p.sources) WHERE value = 'hf')
                 AND NOT EXISTS (SELECT 1 FROM digest_entries d WHERE d.arxiv_id = p.arxiv_id)
               ORDER BY p.hf_upvotes DESC
               LIMIT 1""",
            (run_id, upvote_threshold),
        ).fetchone()
    if row is None:
        return None
    return Store._row_to_paper(row)


def pick_bridging(
    store: Store,
    *,
    run_id: int,
    primary: str,
    model: str,
    min_candidates: int = 10,
    sample_size: int = 30,
) -> tuple[Paper, str] | None:
    """LLM-picked tangential paper with a bridging reason."""
    with store.connect() as conn:
        rows = conn.execute(
            """SELECT p.*, s.score FROM papers p
               JOIN scores s ON s.arxiv_id = p.arxiv_id
               WHERE s.run_id = ?
                 AND s.score >= 3 AND s.score <= 6
                 AND NOT EXISTS (SELECT 1 FROM digest_entries d WHERE d.arxiv_id = p.arxiv_id)""",
            (run_id,),
        ).fetchall()
    if len(rows) < min_candidates:
        log.info("Bridging skipped: %d candidates < %d", len(rows), min_candidates)
        return None

    sample = random.sample(rows, k=min(sample_size, len(rows)))
    candidates = [
        {
            "id": r["arxiv_id"],
            "title": r["title"],
            "abstract": r["abstract"],
            "categories": json.loads(r["categories"]),
            "hf_upvotes": r["hf_upvotes"],
        }
        for r in sample
    ]
    user_msg = (
        f"<user_interests>\n{primary}\n</user_interests>\n\n"
        "From today's papers that our filter rated 3-6 (tangential to the user's "
        "stated interests), pick the ONE that: (a) is highest quality and/or most "
        "important to its own field, AND (b) has a genuine intellectual connection "
        "to the user's interests — a methodological parallel, a shared underlying "
        "problem, or a technique worth stealing. Favor generality and depth over "
        "trendiness.\n\n"
        f"<candidates>\n{json.dumps(candidates, indent=2)}\n</candidates>\n\n"
        'Respond with JSON: {"id": "<arxiv_id>", "bridge_reason": "<two sentences>"}'
    )
    try:
        resp = complete_json_object(
            model=model,
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as e:
        log.warning("Bridging LLM call failed: %s", e)
        return None

    chosen_id = resp.get("id")
    reason = resp.get("bridge_reason", "")
    chosen_row = next((r for r in sample if r["arxiv_id"] == chosen_id), None)
    if chosen_row is None:
        log.warning("Bridging returned unknown id %r", chosen_id)
        return None
    return Store._row_to_paper(chosen_row), reason
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_surprise.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/surprise.py tests/test_surprise.py
git commit -m "add surprise picks: hot-outside-field + bridging"
```

---

## Task 13: `render.py` + Jinja template

**Files:**
- Create: `templates/digest.md.j2`
- Create: `src/recommender/render.py`
- Create: `tests/test_render.py`

- [ ] **Step 1: Create the template**

Create `templates/digest.md.j2`:
```jinja
# ML Papers Digest — {{ date }}

*{{ on_interest|length }} on-interest · {{ surprise_count }} surprise · {{ total_seen }} papers seen · scored by `{{ scoring_model }}`*

---

## On interest

{% for item in on_interest %}
### {{ loop.index }}. {{ item.paper.title }} · score {{ '%.1f' % item.score }}
**{{ item.paper.authors|join(", ") }}** · {{ item.paper.categories|join(", ") }} · [arxiv:{{ item.paper.arxiv_id }}]({{ item.paper.url }}){% if item.paper.hf_upvotes %} · HF ↑{{ item.paper.hf_upvotes }}{% endif %}

> {{ item.paper.abstract|truncate(600) }}

*Why {{ '%.1f' % item.score }}:* {{ item.why }}
*(relevance {{ item.breakdown.relevance }} · quality {{ item.breakdown.quality }} · field-importance {{ item.breakdown.field_importance }})*

---
{% endfor %}

{% if hot_outside is not none %}
## Surprise — hot outside your field
### {{ hot_outside.paper.title }} · HF ↑{{ hot_outside.paper.hf_upvotes }}
**{{ hot_outside.paper.authors|join(", ") }}** · [arxiv:{{ hot_outside.paper.arxiv_id }}]({{ hot_outside.paper.url }})

> {{ hot_outside.paper.abstract|truncate(600) }}

*Trending on HF despite scoring {{ '%.1f' % hot_outside.score }} against your interests.*

---
{% endif %}

{% if bridging is not none %}
## Surprise — bridging pick
### {{ bridging.paper.title }}
**{{ bridging.paper.authors|join(", ") }}** · {{ bridging.paper.categories|join(", ") }} · [arxiv:{{ bridging.paper.arxiv_id }}]({{ bridging.paper.url }})

> {{ bridging.paper.abstract|truncate(600) }}

**Connection:** {{ bridging.bridge_reason }}

---
{% endif %}

*Run: {{ run_duration_s }}s · {% if errors %}errors: {{ errors }}{% else %}no errors{% endif %}*
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_render.py`:
```python
from datetime import datetime, timezone

from recommender.models import Paper
from recommender.render import OnInterestItem, HotOutsideItem, BridgingItem, render_digest


def _paper(arxiv_id, title="Title", upvotes=None):
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="This is an abstract with lots of text.",
        authors=("Alice", "Bob"),
        categories=("cs.LG",),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        sources=("arxiv",),
        hf_upvotes=upvotes,
        first_seen_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


def test_render_includes_all_sections():
    md = render_digest(
        date="2026-04-24",
        on_interest=[
            OnInterestItem(
                paper=_paper("2604.00001", "Foo Paper"),
                score=8.5,
                why="very relevant",
                breakdown={"relevance": 8, "quality": 9, "field_importance": 8},
            ),
        ],
        hot_outside=HotOutsideItem(paper=_paper("2604.00002", "Hot Paper", upvotes=100), score=3.0),
        bridging=BridgingItem(paper=_paper("2604.00003", "Bridge Paper"), bridge_reason="neat parallel"),
        total_seen=42,
        scoring_model="openrouter/anthropic/claude-haiku-4-5",
        run_duration_s=12.3,
        errors="",
    )
    assert "ML Papers Digest — 2026-04-24" in md
    assert "Foo Paper" in md
    assert "score 8.5" in md
    assert "Hot Paper" in md and "HF ↑100" in md
    assert "Bridge Paper" in md and "neat parallel" in md
    assert "scored by `openrouter/anthropic/claude-haiku-4-5`" in md


def test_render_handles_missing_surprises():
    md = render_digest(
        date="2026-04-24",
        on_interest=[OnInterestItem(paper=_paper("2604.00001"), score=7.0, why="r",
                                    breakdown={"relevance": 7, "quality": 7, "field_importance": 7})],
        hot_outside=None,
        bridging=None,
        total_seen=10,
        scoring_model="m",
        run_duration_s=1.0,
        errors="",
    )
    assert "hot outside your field" not in md
    assert "bridging pick" not in md
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement `src/recommender/render.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from recommender.models import Paper

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"


@dataclass(frozen=True)
class OnInterestItem:
    paper: Paper
    score: float
    why: str
    breakdown: dict[str, int]


@dataclass(frozen=True)
class HotOutsideItem:
    paper: Paper
    score: float


@dataclass(frozen=True)
class BridgingItem:
    paper: Paper
    bridge_reason: str


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_digest(
    *,
    date: str,
    on_interest: list[OnInterestItem],
    hot_outside: HotOutsideItem | None,
    bridging: BridgingItem | None,
    total_seen: int,
    scoring_model: str,
    run_duration_s: float,
    errors: str,
) -> str:
    surprise_count = int(hot_outside is not None) + int(bridging is not None)
    tmpl = _env().get_template("digest.md.j2")
    return tmpl.render(
        date=date,
        on_interest=on_interest,
        hot_outside=hot_outside,
        bridging=bridging,
        total_seen=total_seen,
        scoring_model=scoring_model,
        surprise_count=surprise_count,
        run_duration_s=f"{run_duration_s:.1f}",
        errors=errors,
    )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_render.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/recommender/render.py templates/digest.md.j2 tests/test_render.py
git commit -m "add Jinja digest renderer"
```

---

## Task 14: `mail.py` — SMTP send

**Files:**
- Create: `src/recommender/mail.py`
- Create: `tests/test_mail.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mail.py`:
```python
from unittest.mock import MagicMock

from recommender.mail import send


def test_send_calls_smtp_ssl_with_auth_and_multipart(mocker):
    smtp_class = mocker.patch("recommender.mail.smtplib.SMTP_SSL")
    server = MagicMock()
    smtp_class.return_value.__enter__.return_value = server

    send(
        subject="Test subject",
        markdown_body="# Heading\n\nBody text.",
        to_addr="to@example.com",
        from_addr="from@example.com",
        smtp_password="app-password",
        smtp_host="smtp.gmail.com",
        smtp_port=465,
    )

    smtp_class.assert_called_once_with("smtp.gmail.com", 465)
    server.login.assert_called_once_with("from@example.com", "app-password")
    server.send_message.assert_called_once()
    sent_msg = server.send_message.call_args[0][0]
    assert sent_msg["Subject"] == "Test subject"
    assert sent_msg["From"] == "from@example.com"
    assert sent_msg["To"] == "to@example.com"
    # multipart: must contain both plain and html parts
    parts = list(sent_msg.iter_parts()) if sent_msg.is_multipart() else [sent_msg]
    content_types = {p.get_content_type() for p in parts}
    assert "text/plain" in content_types
    assert "text/html" in content_types
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_mail.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/recommender/mail.py`**

```python
from __future__ import annotations

import smtplib
from email.message import EmailMessage

import markdown as md_lib


def send(
    *,
    subject: str,
    markdown_body: str,
    to_addr: str,
    from_addr: str,
    smtp_password: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 465,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(markdown_body)
    html = md_lib.markdown(markdown_body, extensions=["extra"])
    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(from_addr, smtp_password)
        server.send_message(msg)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_mail.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/mail.py tests/test_mail.py
git commit -m "add SMTP email sender"
```

---

## Task 15: `config.py` — Settings dataclass

**Files:**
- Create: `src/recommender/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/recommender/config.py`**

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    scoring_model: str
    bridging_model: str
    batch_size: int

    arxiv_categories: tuple[str, ...]
    arxiv_max_backfill_days: int

    on_interest_min: int
    on_interest_max: int
    on_interest_threshold: float
    hf_upvote_threshold_for_hot_surprise: int

    email_to: str
    email_from: str
    smtp_password: str

    db_path: Path
    digests_dir: Path
    logs_dir: Path
    memory_md: Path
    claude_projects_root: Path

    @classmethod
    def from_env(cls, project_root: Path, env_file: Path | None = None) -> "Settings":
        if env_file is None:
            env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        return cls(
            scoring_model=os.getenv("SCORING_MODEL", "openrouter/anthropic/claude-haiku-4-5"),
            bridging_model=os.getenv("BRIDGING_MODEL", "openrouter/anthropic/claude-haiku-4-5"),
            batch_size=_int("BATCH_SIZE", 20),
            arxiv_categories=tuple(
                os.getenv("ARXIV_CATEGORIES", "cs.LG,cs.AI,cs.CV,cs.CL,stat.ML").split(",")
            ),
            arxiv_max_backfill_days=_int("ARXIV_MAX_BACKFILL_DAYS", 7),
            on_interest_min=_int("ON_INTEREST_MIN", 5),
            on_interest_max=_int("ON_INTEREST_MAX", 15),
            on_interest_threshold=_float("ON_INTEREST_THRESHOLD", 7.0),
            hf_upvote_threshold_for_hot_surprise=_int("HF_UPVOTE_THRESHOLD", 10),
            email_to=_require("EMAIL_TO"),
            email_from=_require("EMAIL_FROM"),
            smtp_password=_require("GMAIL_APP_PASSWORD"),
            db_path=project_root / "data" / "papers.sqlite",
            digests_dir=project_root / "digests",
            logs_dir=project_root / "logs",
            memory_md=project_root / "MEMORY.md",
            claude_projects_root=Path.home() / ".claude" / "projects",
        )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/config.py tests/test_config.py
git commit -m "add Settings config loaded from env"
```

---

## Task 16: `main.py` — pipeline orchestration

**Files:**
- Create: `src/recommender/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_main.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/recommender/main.py`**

```python
from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timedelta, timezone

from recommender import evaluate, interests, mail, render, surprise
from recommender.config import Settings
from recommender.models import DigestEntry
from recommender.render import BridgingItem, HotOutsideItem, OnInterestItem
from recommender.sources import arxiv, huggingface as hf
from recommender.store import Store

log = logging.getLogger(__name__)


def _since(store: Store, max_backfill_days: int) -> datetime:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT MAX(finished_at) FROM runs WHERE status='ok'"
        ).fetchone()
    earliest = datetime.now(timezone.utc) - timedelta(days=max_backfill_days)
    if row and row[0]:
        last = datetime.fromisoformat(row[0])
        return max(last, earliest)
    return earliest


def run_pipeline(
    settings: Settings,
    *,
    force_date: str | None = None,
    dry_run: bool = False,
    no_email: bool = False,
    backfill_days: int | None = None,
) -> None:
    settings.digests_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    store = Store(settings.db_path)
    store.init_db()
    run_id = store.start_run()
    started = time.time()

    try:
        since = (
            datetime.now(timezone.utc) - timedelta(days=backfill_days)
            if backfill_days is not None
            else _since(store, settings.arxiv_max_backfill_days)
        )
        arxiv_papers = arxiv.fetch(categories=settings.arxiv_categories, since=since)
        try:
            hf_papers = hf.fetch()
        except Exception as e:
            log.warning("HF fetch failed: %s", e)
            hf_papers = []

        all_papers = arxiv_papers + hf_papers
        store.upsert_papers(all_papers)
        needing = store.papers_needing_scoring(run_id)

        primary, secondary = interests.load(
            memory_md=settings.memory_md,
            claude_projects_root=settings.claude_projects_root,
        )
        scores = evaluate.score_papers(
            needing,
            primary=primary,
            secondary=secondary,
            run_id=run_id,
            model=settings.scoring_model,
            batch_size=settings.batch_size,
        )
        store.save_scores(scores)

        digest_date = force_date or datetime.now(timezone.utc).date().isoformat()
        top = store.top_scoring_today(
            run_id,
            threshold=settings.on_interest_threshold,
            min_count=settings.on_interest_min,
            max_count=settings.on_interest_max,
        )
        on_interest = [
            _on_interest_item(store, p, run_id) for p in top
        ]
        hot = surprise.pick_hot_outside_field(
            store, run_id=run_id,
            upvote_threshold=settings.hf_upvote_threshold_for_hot_surprise,
        )
        hot_item = _hot_item(store, hot, run_id) if hot else None
        bridging_pair = surprise.pick_bridging(
            store, run_id=run_id, primary=primary,
            model=settings.bridging_model,
        )
        bridging_item = (
            BridgingItem(paper=bridging_pair[0], bridge_reason=bridging_pair[1])
            if bridging_pair else None
        )

        md = render.render_digest(
            date=digest_date,
            on_interest=on_interest,
            hot_outside=hot_item,
            bridging=bridging_item,
            total_seen=len(all_papers),
            scoring_model=settings.scoring_model,
            run_duration_s=time.time() - started,
            errors="",
        )
        digest_path = settings.digests_dir / f"{digest_date}.md"
        digest_path.write_text(md)

        if not dry_run:
            entries: list[DigestEntry] = []
            for rank, item in enumerate(on_interest, start=1):
                entries.append(DigestEntry(digest_date, item.paper.arxiv_id, "on_interest", rank))
            if hot_item is not None:
                entries.append(DigestEntry(digest_date, hot_item.paper.arxiv_id, "surprise_hot", 1))
            if bridging_item is not None:
                entries.append(DigestEntry(digest_date, bridging_item.paper.arxiv_id, "surprise_bridge", 1))
            store.mark_sent(entries)

            if not no_email:
                subj = (
                    f"ML digest — {digest_date} · "
                    f"{len(on_interest)}+{int(hot_item is not None) + int(bridging_item is not None)} papers"
                )
                mail.send(
                    subject=subj,
                    markdown_body=md,
                    to_addr=settings.email_to,
                    from_addr=settings.email_from,
                    smtp_password=settings.smtp_password,
                )

        store.record_run(
            run_id,
            status="ok",
            papers_seen=len(all_papers),
            papers_scored=len(scores),
            digest_date=digest_date,
        )
    except Exception as e:
        log.exception("Pipeline failed")
        store.record_run(run_id, status="error", error=traceback.format_exc())
        raise


def _on_interest_item(store: Store, paper, run_id: int) -> OnInterestItem:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT score, justification FROM scores WHERE arxiv_id = ? AND run_id = ?",
            (paper.arxiv_id, run_id),
        ).fetchone()
    payload = json.loads(row["justification"])
    return OnInterestItem(
        paper=paper,
        score=row["score"],
        why=payload.get("why", ""),
        breakdown=payload.get("breakdown", {"relevance": 0, "quality": 0, "field_importance": 0}),
    )


def _hot_item(store: Store, paper, run_id: int) -> HotOutsideItem:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT score FROM scores WHERE arxiv_id = ? AND run_id = ?",
            (paper.arxiv_id, run_id),
        ).fetchone()
    return HotOutsideItem(paper=paper, score=row["score"])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_main.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/main.py tests/test_main.py
git commit -m "add pipeline orchestration in main.py"
```

---

## Task 17: `__main__.py` — CLI entry with argparse

**Files:**
- Create: `src/recommender/__main__.py`
- Modify: `tests/test_main.py` (add CLI tests)

- [ ] **Step 1: Add failing CLI tests**

Append to `tests/test_main.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_main.py -v -k cli`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/recommender/__main__.py`**

```python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from recommender.config import Settings
from recommender.main import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recommender")
    parser.add_argument("--dry-run", action="store_true", help="Write markdown but skip email and digest_entries persistence")
    parser.add_argument("--no-email", action="store_true", help="Persist digest but skip email")
    parser.add_argument("--backfill", type=int, default=None, help="Look back N days instead of since-last-run")
    parser.add_argument("--force-date", type=str, default=None, help="Regenerate the digest for YYYY-MM-DD")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_env(project_root=Path.cwd())
    run_pipeline(
        settings,
        force_date=args.force_date,
        dry_run=args.dry_run,
        no_email=args.no_email,
        backfill_days=args.backfill,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_main.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommender/__main__.py tests/test_main.py
git commit -m "add CLI entry with argparse flags"
```

---

## Task 18: Starter `MEMORY.md` + `README.md` + full test sweep

**Files:**
- Create: `MEMORY.md`
- Create: `README.md`

- [ ] **Step 1: Create starter `MEMORY.md`**

```markdown
# Research interests

## Core interests
- <replace with your actual core interests>

## Currently working on
- <1-3 bullets about active projects>

## Authors I follow
- <optional>

## Venues I care about
- NeurIPS, ICML, ICLR, COLM, TMLR

## Keywords to boost
- <optional>

## Not interested in
- <optional>
```

- [ ] **Step 2: Create `README.md`**

```markdown
# paper-recommender

Single-user daily ML paper digest. Scrapes arXiv + Hugging Face Daily Papers,
scores against a hand-curated `MEMORY.md`, and emails a markdown digest.

## Setup

1. Install `uv` if needed: https://docs.astral.sh/uv/
2. `uv sync --extra dev`
3. Copy `.env.example` to `.env` and fill in:
   - At least one of `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`
   - `SCORING_MODEL` and `BRIDGING_MODEL` (LiteLLM format, e.g. `openrouter/anthropic/claude-haiku-4-5`)
   - `EMAIL_TO`, `EMAIL_FROM`
   - `GMAIL_APP_PASSWORD` — see "Gmail app password" below
4. Edit `MEMORY.md` to describe your research interests.

### Gmail app password

1. Enable 2FA on the Gmail account.
2. Visit https://myaccount.google.com/apppasswords
3. Create a password for "Mail / Other: paper-recommender". Paste into `.env` as `GMAIL_APP_PASSWORD`.

## Usage

```bash
# Manual run
uv run python -m recommender

# Dry run: writes the markdown digest, skips email and digest_entries
uv run python -m recommender --dry-run

# Regenerate a specific day (overwrites digests/YYYY-MM-DD.md, remails)
uv run python -m recommender --force-date 2026-04-24

# Look back N days instead of since-last-run
uv run python -m recommender --backfill 3
```

### Cron

Add to your crontab (`crontab -e`):

```cron
0 7 * * * cd /home/YOU/src/paper-recommender && /usr/bin/env -S uv run python -m recommender >> logs/cron.log 2>&1
```

## Tests

```bash
uv run pytest
```
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add MEMORY.md README.md
git commit -m "add starter MEMORY.md and README"
```

---

## Post-plan verification checklist

After all tasks are complete, confirm end-to-end with a manual sanity run:

- [ ] Fill in a real `MEMORY.md` and `.env`.
- [ ] Run `uv run python -m recommender --dry-run` and open `digests/<today>.md`. Verify rubric breakdowns are present and the three sections render as expected.
- [ ] Run `uv run python -m recommender` (not dry) and confirm the email arrives.
- [ ] Add the cron entry and verify it fires the next morning via `logs/cron.log` and the `runs` table.

---

## Self-review notes

- **Spec coverage:**
  - MEMORY.md format — Task 18 (starter) + `interests.load` in Task 9.
  - Multi-source scrape — Tasks 7 (arXiv) + 8 (HF).
  - SQLite schema — Task 3; upsert/dedup Task 4; scores/run lifecycle Task 5; digest pickers Task 6.
  - LiteLLM abstraction + Anthropic `cache_control` — Task 10.
  - Rubric (0-2 unimportant, 3-5 tangential, 6-10 relevant) with three factors — Task 11 (`SCORING_RUBRIC`).
  - Surprise (hot-outside-field + bridging) — Task 12.
  - Markdown digest — Task 13.
  - SMTP email — Task 14.
  - Config — Task 15.
  - Pipeline + failure isolation — Task 16.
  - CLI flags (`--dry-run`, `--force-date`, `--backfill`, `--no-email`) — Task 17.
  - Cron wiring — Task 18 (README).
- **Placeholder scan:** no TBDs, no "TODO", no "similar to Task N". Each step has complete code or a concrete command.
- **Type consistency:** `Paper`/`Score`/`DigestEntry` defined once (Task 2), `Store` methods signatures consistent across Tasks 3-6 and usage in Tasks 11-16. Render items (`OnInterestItem`, `HotOutsideItem`, `BridgingItem`) defined in Task 13 and used in Task 16.
- **Scope:** one implementation plan; no sub-systems leaked in.
