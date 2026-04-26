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
    upvote_threshold: int,
) -> Paper | None:
    """Highest-upvoted HF paper whose latest score is <5 and is not yet digested."""
    return store.hot_outside_field_pick(upvote_threshold=upvote_threshold)


def pick_bridging(
    store: Store,
    *,
    primary: str,
    model: str,
    min_candidates: int = 10,
    sample_size: int = 30,
) -> tuple[Paper, str] | None:
    """LLM-picked tangential paper with a bridging reason."""
    candidates = store.bridging_candidates()
    if len(candidates) < min_candidates:
        log.info("Bridging skipped: %d candidates < %d", len(candidates), min_candidates)
        return None

    sample = random.sample(candidates, k=min(sample_size, len(candidates)))
    items = [
        {
            "id": p.arxiv_id,
            "title": p.title,
            "abstract": p.abstract,
            "categories": list(p.categories),
            "hf_upvotes": p.hf_upvotes,
        }
        for p in sample
    ]
    user_msg = (
        f"<user_interests>\n{primary}\n</user_interests>\n\n"
        "From today's papers that our filter rated 3-6 (tangential to the user's "
        "stated interests), pick the ONE that: (a) is highest quality and/or most "
        "important to its own field, AND (b) has a genuine intellectual connection "
        "to the user's interests — a methodological parallel, a shared underlying "
        "problem, or a technique worth stealing. Favor generality and depth over "
        "trendiness.\n\n"
        f"<candidates>\n{json.dumps(items, indent=2)}\n</candidates>\n\n"
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
    chosen = next((p for p in sample if p.arxiv_id == chosen_id), None)
    if chosen is None:
        log.warning("Bridging returned unknown id %r", chosen_id)
        return None
    return chosen, reason
