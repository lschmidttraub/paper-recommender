from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone

import feedparser
import requests

from recommender.models import Paper

log = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_VERSION_SUFFIX = re.compile(r"v\d+$")
_ID_FROM_URL = re.compile(r"arxiv\.org/abs/([^/]+?)(?:v\d+)?$")

# arXiv asks API clients to send a descriptive User-Agent and to back off when
# throttled. Default python-requests UAs and bare retries get 429'd hard,
# especially from shared cloud IPs (e.g. GitHub Actions runners).
_USER_AGENT = (
    "paper-recommender/0.1 (+https://github.com/lschmidttraub/paper-recommender)"
)
_RETRY_STATUSES = frozenset({429, 503})


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
    max_retries: int = 5,
    backoff_base: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Paper]:
    until = until or datetime.now(timezone.utc)
    url = build_query(categories, since, until)
    sess = session or requests.Session()
    headers = {"User-Agent": _USER_AGENT}

    resp = None
    for attempt in range(max_retries + 1):
        resp = sess.get(url, timeout=60, headers=headers)
        if resp.status_code not in _RETRY_STATUSES:
            break
        if attempt == max_retries:
            break
        delay = _retry_delay(resp, attempt, backoff_base)
        log.warning(
            "arXiv returned %s; retrying in %.0fs (attempt %d/%d)",
            resp.status_code, delay, attempt + 1, max_retries,
        )
        sleep(delay)

    resp.raise_for_status()
    return parse_atom(resp.text, now=datetime.now(timezone.utc))


def _retry_delay(resp: requests.Response, attempt: int, backoff_base: float) -> float:
    """Honor a numeric Retry-After header; otherwise exponential backoff."""
    retry_after = resp.headers.get("Retry-After", "")
    if retry_after.strip().isdigit():
        return float(retry_after)
    return backoff_base * (2 ** attempt)


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
