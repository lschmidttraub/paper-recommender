from datetime import datetime, timezone

from recommender.models import DigestEntry, Paper, Score


def test_paper_is_frozen():
    p = Paper(
        arxiv_id="2604.00001",
        title="A Paper",
        abstract="abstract text",
        authors=("Alice", "Bob"),
        categories=("cs.LG",),
        url="https://arxiv.org/abs/2604.00001",
        published_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        sources=("arxiv",),
        hf_upvotes=None,
        first_seen_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )
    try:
        p.title = "changed"  # type: ignore[misc]
    except Exception as e:
        assert "frozen" in str(e).lower() or "cannot assign" in str(e).lower()
    else:
        raise AssertionError("Paper should be frozen")


def test_score_has_breakdown_and_why():
    s = Score(
        arxiv_id="2604.00001",
        run_id=1,
        model="openrouter/anthropic/claude-haiku-4-5",
        score=8.5,
        breakdown={"relevance": 7, "quality": 9, "field_importance": 9},
        why="Landmark work on X",
        scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )
    assert s.score == 8.5
    assert s.breakdown["quality"] == 9


def test_digest_entry_holds_section_and_rank():
    e = DigestEntry(digest_date="2026-04-24", arxiv_id="2604.00001", section="on_interest", rank=1)
    assert e.section == "on_interest"
    assert e.rank == 1
