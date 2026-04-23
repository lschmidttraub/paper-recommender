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
