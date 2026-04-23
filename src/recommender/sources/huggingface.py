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
