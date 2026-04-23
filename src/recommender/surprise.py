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
