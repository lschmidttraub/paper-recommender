# Paper Recommender — Design Spec

**Date:** 2026-04-24
**Status:** Approved for planning
**Owner:** Leo Schmidt-Traub

## Goal

A personal daily ML paper digest. Each morning, scrape new papers from arXiv and Hugging Face Daily Papers, score them against a hand-curated `MEMORY.md` describing the user's research interests, and deliver a ranked digest by email plus a local markdown file. Include a small "surprise" section so genuinely important or interesting work outside the user's stated interests still surfaces.

Single-user, local, runs from cron.

## Non-goals (v1)

- Feedback loop (click/star/dismiss tracking). Design leaves room to add later.
- Multi-user or profile support.
- Full-text PDF ingestion. Abstracts only.
- Web UI or dashboard.
- Citation-based re-ranking or additional sources (OpenReview, Semantic Scholar).
- CI or deployment infrastructure.

## High-level architecture

```
cron ─► python -m recommender
         │
         ├─► sources/{arxiv,huggingface} ──► list[Paper]
         ├─► store.upsert_papers           ──► SQLite (papers)
         ├─► interests.load                ──► MEMORY.md + Claude Code memory scan
         ├─► evaluate.score_batches        ──► LiteLLM (cached interests block)
         ├─► store.save_scores             ──► SQLite (scores)
         ├─► pick digest
         │     ├─ on_interest     : SQL rank by score
         │     ├─ surprise_hot    : SQL on hf_upvotes + low score
         │     └─ surprise_bridge : LLM call on tangential candidates
         ├─► render.digest (Jinja2)        ──► digests/YYYY-MM-DD.md
         ├─► mail.send (SMTP/Gmail)        ──► user's inbox
         └─► store.mark_sent + record_run  ──► SQLite (digest_entries, runs)
```

## Project layout

```
paper-recommender/
├── MEMORY.md                    # user's canonical interests (hand-edited)
├── pyproject.toml               # uv + deps
├── uv.lock
├── .env                         # secrets + config overrides
├── .env.example                 # committed template
├── README.md                    # setup, Gmail app-password instructions, cron snippet
├── src/recommender/
│   ├── __init__.py
│   ├── __main__.py              # python -m recommender entrypoint
│   ├── config.py                # Settings dataclass, loaded from env
│   ├── models.py                # Paper, Score, DigestEntry dataclasses
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── arxiv.py             # fetch + parse arXiv ATOM feed
│   │   └── huggingface.py       # fetch HF Daily Papers + upvotes
│   ├── store.py                 # SQLite schema, migrations, all DB access
│   ├── interests.py             # MEMORY.md loader + Claude Code memory scan
│   ├── llm.py                   # thin LiteLLM wrapper (only importer of litellm)
│   ├── evaluate.py              # batched scoring with cached interest block
│   ├── surprise.py              # hot-outside-field + bridging picks
│   ├── render.py                # Jinja2 markdown digest renderer
│   ├── mail.py                  # smtplib send (text + HTML)
│   └── main.py                  # orchestrates the pipeline
├── templates/
│   └── digest.md.j2
├── digests/                     # YYYY-MM-DD.md files, one per run
├── data/
│   └── papers.sqlite            # all state
├── logs/
│   └── YYYY-MM-DD.log
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── arxiv_sample.atom
    │   └── hf_daily_sample.json
    ├── test_arxiv.py
    ├── test_huggingface.py
    ├── test_store.py
    ├── test_interests.py
    ├── test_evaluate.py
    ├── test_surprise.py
    └── test_render.py
```

### Module boundaries and responsibilities

- `sources/*` only fetch + normalize into `Paper` dataclass. No scoring, no DB.
- `store.py` owns all SQL. Nothing else imports `sqlite3`.
- `llm.py` is the only file that imports `litellm`. Swapping the LLM abstraction later is a one-file change.
- `evaluate.py` only turns papers + interests into scores. No I/O beyond the LLM client.
- `main.py` is the only place the pipeline is stitched together end-to-end.

## Data flow — one daily run

1. **Load interests.** `interests.load()` reads `MEMORY.md` verbatim and scans every `~/.claude/projects/*/memory/MEMORY.md` + `project_*.md` / `user_*.md` within those dirs. Concatenates Claude Code content under project headers, returns `(primary_interests_text, secondary_signals_text)`.
2. **Scrape new papers.**
   - `arxiv.fetch(since, categories)` — ATOM feed for each of `cs.LG, cs.AI, cs.CV, cs.CL, stat.ML`. `since` is the last successful run's `finished_at` from the `runs` table; clamped to `arxiv_max_backfill_days` (default 7).
   - `huggingface.fetch_daily_papers()` — HF Daily Papers API. Each item carries `upvotes` and a linked arXiv ID.
   - Results merged into a single `list[Paper]`; HF-only papers that link to an arXiv ID merge into the same canonical paper.
3. **Persist + dedup.** `store.upsert_papers(papers)` inserts only new `arxiv_id`s (existing rows update `sources` and `hf_upvotes` when HF catches up). `store.papers_needing_scoring(run_id)` returns papers without a score for the current run.
4. **Score in batches.** For each batch of `batch_size` (20) papers, `llm.score_batch(batch, primary, secondary, rubric)` returns `[{id, score, breakdown, why}]`. Failures retry once; on second failure, log and skip those papers (no `scores` row → retried on the next run).
5. **Pick digest.**
   - `on_interest`: `store.top_scoring_today(run_id, min_score=7, min_count=5, max_count=15)` — excludes any paper with an existing `digest_entries` row.
   - `surprise_hot`: single SQL query; highest HF-upvoted paper from today's batch with score < 5 and `hf_upvotes >= 10`. Skip slot if nothing qualifies.
   - `surprise_bridge`: LLM call on up to 30 papers with `3 <= score <= 6`, asking for the one most likely to genuinely interest the user via an intellectual bridge. Skip slot if fewer than 10 candidates.
6. **Render.** `render.digest(date, on_interest, surprise_hot, surprise_bridge, run_meta)` → markdown file at `digests/YYYY-MM-DD.md`.
7. **Email.** `mail.send(subject, markdown_body, to_addr)` — SMTPS to `smtp.gmail.com:465` with Gmail app password. Sends both `text/plain` (markdown source) and `text/html` (rendered via `markdown` library).
8. **Finalize.** `store.mark_sent(paper_ids, date, section, rank)` writes `digest_entries`; `store.record_run(status, counts)` marks the run complete.

### Idempotency & failure isolation

- Steps 3, 4, and 8 each commit partial state, so a failure mid-run doesn't lose upstream work.
- Rerunning the same day is safe: papers already scored won't be re-scored; papers already in `digest_entries` won't appear again. Use `--force-date YYYY-MM-DD` to regenerate a specific day's digest.
- Email failures preserve the markdown file — retry with `--force-date ... --email-only` (implementation detail: achievable because the markdown is the source of truth).

## Data artifacts

### `MEMORY.md` format

Fixed section headers, all optional:

```markdown
# Research interests

## Core interests
- <bullet list>

## Currently working on
- <bullet list>

## Authors I follow
- <bullet list>

## Venues I care about
- <bullet list>

## Keywords to boost
- <bullet list>

## Not interested in
- <bullet list>
```

`interests.py` does not parse beyond reading the file. The whole text is passed to the LLM under `<user_interests>`. Editing takes effect next run.

### Claude Code memory scan (secondary signal)

`interests.py` globs `~/.claude/projects/*/memory/MEMORY.md` and the `project_*.md` / `user_*.md` files in those directories. Concatenates under project headers. Passed to the LLM under `<secondary_signals>` — explicitly lower-weight than `<user_interests>`.

### SQLite schema (`data/papers.sqlite`)

```sql
CREATE TABLE papers (
    arxiv_id      TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    abstract      TEXT NOT NULL,
    authors       TEXT NOT NULL,   -- JSON array
    categories    TEXT NOT NULL,   -- JSON array
    url           TEXT NOT NULL,
    published_at  TEXT NOT NULL,   -- ISO8601
    sources       TEXT NOT NULL,   -- JSON array e.g. ["arxiv","hf"]
    hf_upvotes    INTEGER,         -- NULL if not on HF
    first_seen_at TEXT NOT NULL
);
CREATE INDEX idx_papers_published ON papers(published_at);

CREATE TABLE scores (
    arxiv_id      TEXT NOT NULL,
    run_id        INTEGER NOT NULL,
    model         TEXT NOT NULL,
    score         REAL NOT NULL,
    justification TEXT NOT NULL,   -- JSON {why, breakdown:{relevance,quality,field_importance}}
    scored_at     TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, run_id),
    FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id),
    FOREIGN KEY (run_id)   REFERENCES runs(run_id)
);

CREATE TABLE digest_entries (
    digest_date   TEXT NOT NULL,
    arxiv_id      TEXT NOT NULL,
    section       TEXT NOT NULL,   -- 'on_interest' | 'surprise_hot' | 'surprise_bridge'
    rank          INTEGER NOT NULL,
    PRIMARY KEY (digest_date, arxiv_id)
);
CREATE INDEX idx_digest_arxiv ON digest_entries(arxiv_id);

CREATE TABLE runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,   -- 'running' | 'ok' | 'error'
    papers_seen   INTEGER,
    papers_scored INTEGER,
    digest_date   TEXT,
    error         TEXT
);
```

Invariants:
- `arxiv_id` is canonical; HF papers that reference an arXiv ID merge into the existing row.
- A paper is sent in a digest at most once, ever (guaranteed by `digest_entries` uniqueness + `top_scoring_today` excluding existing entries).
- `scores` allows multiple rows per paper across runs (for re-scoring when interests shift).

## LLM evaluation

### Multi-provider layer

All LLM calls go through `llm.py`, which wraps `litellm.completion`. Model selection is config-driven:

```python
# examples:
SCORING_MODEL  = "openrouter/anthropic/claude-haiku-4-5"
SCORING_MODEL  = "openai/gpt-4o-mini"
SCORING_MODEL  = "gemini/gemini-2.0-flash"
SCORING_MODEL  = "anthropic/claude-haiku-4-5"
BRIDGING_MODEL = "openrouter/anthropic/claude-haiku-4-5"
```

OpenRouter is a first-class target (per user preference); LiteLLM handles routing. Provider API keys (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) are read from env by LiteLLM.

`llm.py` also:
- Applies `cache_control` markers on the cacheable block when the underlying provider is Anthropic (via LiteLLM's pass-through), so the ~2-5k-token interests block costs cache-read rates for batches 2..N.
- Normalizes JSON output: requests JSON mode where supported; falls back to strict-parse → regex-extract array → retry on parse failure.

### Scoring prompt

Three-part layout so cacheable content is the prefix:

1. **System** (short, static): "You are a research assistant ranking ML papers for a specific user. Return only valid JSON."
2. **Cacheable block** (identical across a run):
   - `<user_interests>` — MEMORY.md verbatim
   - `<secondary_signals>` — Claude Code memory scan
   - `<scoring_rubric>` — see below
3. **Per-batch**: up to 20 papers as JSON (`id`, `title`, `abstract`, `categories`, `hf_upvotes`). Expects JSON array response.

### Scoring rubric (in the cacheable block)

```
Score each paper 0-10 as a JOINT assessment of three factors:
  (a) RELEVANCE        — match to <user_interests>
  (b) QUALITY          — novelty, methodological rigor, specificity of claims,
                         clarity of contribution (inferred from abstract)
  (c) FIELD IMPORTANCE — potential to be a landmark/foundational paper even
                         in a field the user did not list. Signals: bold
                         novel claims, new paradigms, author track record,
                         community attention (hf_upvotes).

Band guide:
  0-2  Unimportant: weak AND off-topic / low general merit
  3-5  Tangential: off-topic with some merit, OR on-topic but mediocre
  6-7  Solidly relevant + reasonable quality, OR important in nearby field
  8-9  Strongly relevant + good quality, OR a landmark-level paper in any field
  10   Exceptional: generational in user's field, OR field-wide landmark
       that connects to the user's interests

Apply the "Not interested in" section as a hard signal — deduct 3+ points,
UNLESS the paper is genuinely landmark-level (≥8), in which case field
importance wins.
```

### Per-paper response schema

```json
{
  "id": "2604.12345",
  "score": 8.5,
  "breakdown": {"relevance": 6, "quality": 9, "field_importance": 9},
  "why": "Landmark-looking result on X that connects to user's Y interest via..."
}
```

`breakdown` and `why` are stored as JSON in `scores.justification` and surfaced in the digest for transparency.

### Batching

- `batch_size` = 20 papers per call (~8k per-batch tokens + cached ~3k interest block).
- Sequential calls (~15 calls for a typical 300-paper day).
- Parse robustness: strict JSON → regex array extract → retry once → skip on final failure.

### Surprise picks

**Hot outside field** — pure SQL (no LLM):
```sql
SELECT p.arxiv_id FROM papers p
JOIN scores s USING (arxiv_id)
WHERE s.run_id = :current_run
  AND json_extract(p.sources, '$') LIKE '%hf%'
  AND p.hf_upvotes >= :threshold
  AND s.score < 5
  AND NOT EXISTS (SELECT 1 FROM digest_entries d WHERE d.arxiv_id = p.arxiv_id)
ORDER BY p.hf_upvotes DESC
LIMIT 1;
```
Skip the slot if nothing qualifies.

**Bridging** — single LLM call:

```
<user_interests>{{ MEMORY.md }}</user_interests>

From today's ~30 papers rated 3-6 (tangential to the user's stated interests),
pick the ONE that:
  (a) is highest quality and/or most important to its own field, AND
  (b) has a genuine intellectual connection to the user's interests —
      a methodological parallel, a shared underlying problem, or a technique
      worth stealing.

Favor generality and depth over trendiness. A landmark-quality paper from an
adjacent field beats a hot-but-shallow paper in a distant field.

<candidates>[... up to 30 papers with id/title/abstract/categories/hf_upvotes ...]</candidates>

Respond: {"id": "...", "bridge_reason": "..."}  (reason: 2 sentences)
```

Skip the slot if fewer than 10 candidates available.

## Delivery

### Markdown digest

Rendered via Jinja2 template (`templates/digest.md.j2`). Layout per section:

- **Header**: date, counts (on-interest / surprise), total papers seen, scoring model.
- **On-interest section**: numbered items, each with title, authors, categories, arXiv link, HF upvotes (if any), abstract truncated to ~600 chars, `why` justification, and `(relevance · quality · field_importance)` breakdown.
- **Surprise — hot outside your field**: single paper, HF upvotes prominent, same layout minus breakdown.
- **Surprise — bridging pick**: single paper, with `bridge_reason` prominent.
- **Footer**: run duration, errors (if any).

Subject line: `ML digest — {date} · {n_on_interest}+{n_surprise} papers`.

### Email

`smtplib` + `email.mime` — no extra deps. SMTPS (port 465) to `smtp.gmail.com`. Auth: `GMAIL_USER` + `GMAIL_APP_PASSWORD` from env (Gmail 2FA app password; README documents setup). Multipart message with both `text/plain` (markdown) and `text/html` (rendered via the `markdown` library with `extensions=['extra']`).

### Automation

Cron entry (daily 07:00 local):

```cron
0 7 * * *  cd /home/lschmidt-traub/src/paper-recommender && /usr/bin/env -S uv run python -m recommender >> logs/cron.log 2>&1
```

CLI flags on `python -m recommender`:
- `--dry-run` — do not send email, do not write `digest_entries`. Just write markdown.
- `--no-email` — write markdown + persist digest_entries, skip send.
- `--backfill N` — look back N days instead of "since last run."
- `--force-date YYYY-MM-DD` — regenerate a specific day's digest (overwrites existing `digests/YYYY-MM-DD.md`).

Exits nonzero on unrecoverable failure.

## Configuration

```python
@dataclass(frozen=True)
class Settings:
    # LLM
    scoring_model:  str = "openrouter/anthropic/claude-haiku-4-5"
    bridging_model: str = "openrouter/anthropic/claude-haiku-4-5"
    batch_size:     int = 20

    # Sources
    arxiv_categories:        tuple[str, ...] = ("cs.LG", "cs.AI", "cs.CV", "cs.CL", "stat.ML")
    arxiv_max_backfill_days: int = 7

    # Digest sizing
    on_interest_min:                      int   = 5
    on_interest_max:                      int   = 15
    on_interest_threshold:                float = 7.0
    hf_upvote_threshold_for_hot_surprise: int   = 10

    # Delivery (from env)
    email_to:      str
    email_from:    str
    smtp_password: str   # never logged

    # Paths (derived from project root)
    db_path:     Path
    digests_dir: Path
    logs_dir:    Path
    memory_md:   Path
```

`.env` carries secrets + overrides. `.env.example` is committed.

## Error handling

| Failure                    | Behavior                                                                |
| -------------------------- | ----------------------------------------------------------------------- |
| arXiv fetch fails          | Raise; run aborts. Next cron run retries. Nothing corrupted.            |
| HF fetch fails             | Log warning, continue (enrichment, not blocking).                       |
| LLM scoring batch fails    | Retry once. On second failure, skip those papers (no scores row → retry next run). |
| LLM JSON parse fails       | Strict → regex-extract → retry batch.                                    |
| Bridging LLM call fails    | Log, skip the bridging slot. Digest still ships.                        |
| DB locked / IO error       | Raise; run aborts.                                                      |
| Email send fails           | Markdown is already written. Log + raise. Recover with `--force-date`.  |

Every run writes a `runs` row with status + traceback. Python `logging`: INFO to `logs/YYYY-MM-DD.log`, WARNING+ to stderr (cron captures).

## Testing

Unit tests with `pytest`:

- `test_arxiv.py` — parses fixture ATOM feed, asserts `Paper` normalization.
- `test_huggingface.py` — parses fixture JSON, asserts upvote + arXiv-link merge.
- `test_store.py` — in-memory SQLite; upsert idempotency, `papers_needing_scoring`, `top_scoring_today`, `mark_sent`, run logging.
- `test_interests.py` — tmpdir-based Claude Code memory scan + `MEMORY.md` load.
- `test_evaluate.py` — mocks `llm.complete`; asserts batching, prompt structure contains cacheable block, JSON parse fallbacks.
- `test_surprise.py` — in-memory DB with synthetic scores; asserts hot-outside-field SQL and bridging candidate selection.
- `test_render.py` — renders template against fixture, snapshot compare.

No integration tests against real arXiv/HF/LLMs; those are manual `--dry-run` experiments. CI out of scope for v1.

## Runtime & dependencies

- **Python 3.11+**.
- **`uv`** for dependency management (`pyproject.toml` + `uv.lock`).
- Runtime deps: `litellm`, `feedparser` (arXiv ATOM), `requests` (HF), `jinja2`, `markdown`, `python-dotenv`.
- Dev deps: `pytest`, `pytest-mock`.
- No Docker, no services, no queue.

## Open questions (none blocking implementation)

- Whether to eventually add a Gmail reply-based feedback loop ("reply `+` to this digest item to star it"). Left for v2.
- Whether authors from `Authors I follow` should be a post-ranking override (always include) or remain a soft boost. Current design uses soft boost; revisit if dissatisfaction emerges.
