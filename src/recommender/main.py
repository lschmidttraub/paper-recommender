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
        needing = store.papers_without_scores()

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
