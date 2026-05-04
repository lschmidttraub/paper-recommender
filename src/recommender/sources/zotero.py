from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

_API_BASE = "https://api.zotero.org"
_PAGE_LIMIT = 100
_DEFAULT_MAX_ITEMS = 500


@dataclass(frozen=True)
class ZoteroItem:
    key: str
    title: str
    creators: tuple[str, ...]
    year: str
    tags: tuple[str, ...]
    item_type: str
    date_added: str  # ISO8601 from Zotero


def fetch_items(
    api_key: str,
    user_id: str,
    *,
    max_items: int = _DEFAULT_MAX_ITEMS,
    session: requests.Session | None = None,
) -> list[ZoteroItem]:
    """Fetch up to max_items most-recently-added items from the user's Zotero library.

    Excludes attachments and notes (we want bibliographic items). Sorted newest first.
    """
    sess = session or requests.Session()
    headers = {"Zotero-API-Key": api_key, "Zotero-API-Version": "3"}
    items: list[ZoteroItem] = []
    start = 0
    while len(items) < max_items:
        page_limit = min(_PAGE_LIMIT, max_items - len(items))
        url = f"{_API_BASE}/users/{user_id}/items"
        params = {
            "start": start,
            "limit": page_limit,
            "sort": "dateAdded",
            "direction": "desc",
            "itemType": "-attachment || note",
        }
        resp = sess.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        items.extend(_parse_item(entry) for entry in page if _parse_item(entry))
        if len(page) < page_limit:
            break
        start += page_limit
    return items[:max_items]


def _parse_item(entry: dict) -> ZoteroItem | None:
    data = entry.get("data") or {}
    if not data.get("title"):
        return None
    creators = tuple(
        _creator_name(c)
        for c in data.get("creators", [])
        if _creator_name(c)
    )
    tags = tuple(t.get("tag", "") for t in data.get("tags", []) if t.get("tag"))
    return ZoteroItem(
        key=data.get("key", entry.get("key", "")),
        title=data.get("title", "").strip(),
        creators=creators,
        year=_year_from_date(data.get("date", "")),
        tags=tags,
        item_type=data.get("itemType", ""),
        date_added=data.get("dateAdded", ""),
    )


def _creator_name(c: dict) -> str:
    last = (c.get("lastName") or "").strip()
    first = (c.get("firstName") or "").strip()
    name = (c.get("name") or "").strip()
    if last and first:
        return f"{first} {last}"
    return last or name or first


def _year_from_date(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return ""


def format_library(items: list[ZoteroItem]) -> str:
    """Render the library as a compact text block for the LLM cacheable prefix.

    No abstracts (token budget). Title, year, creators, tags, item_type only.
    """
    if not items:
        return ""
    lines = []
    for it in items:
        creators = ", ".join(it.creators[:5])
        if len(it.creators) > 5:
            creators += " et al."
        tags = ", ".join(it.tags[:8])
        year = f" ({it.year})" if it.year else ""
        creators_part = f" — {creators}" if creators else ""
        tags_part = f" [tags: {tags}]" if tags else ""
        lines.append(f"- {it.title}{year}{creators_part}{tags_part}")
    return "\n".join(lines)
