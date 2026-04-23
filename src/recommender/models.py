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
