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
