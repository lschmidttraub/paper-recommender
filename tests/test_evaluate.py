from datetime import datetime, timezone

import pytest

from recommender.evaluate import SCORING_RUBRIC, build_cacheable_prefix, build_user_message, score_papers
from recommender.models import Paper


def _paper(arxiv_id: str, hf_upvotes: int | None = None) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        abstract="Abstract text.",
        authors=("Alice",),
        categories=("cs.LG",),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        sources=("arxiv",),
        hf_upvotes=hf_upvotes,
        first_seen_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


def test_rubric_mentions_three_factors_and_bands():
    assert "RELEVANCE" in SCORING_RUBRIC
    assert "QUALITY" in SCORING_RUBRIC
    assert "FIELD IMPORTANCE" in SCORING_RUBRIC
    assert "0-2" in SCORING_RUBRIC and "3-5" in SCORING_RUBRIC
    assert "10" in SCORING_RUBRIC


def test_build_cacheable_prefix_contains_interests_and_rubric():
    prefix = build_cacheable_prefix(primary="MY INTERESTS", secondary="SECONDARY")
    assert "<user_interests>" in prefix
    assert "MY INTERESTS" in prefix
    assert "<secondary_signals>" in prefix
    assert "SECONDARY" in prefix
    assert "<scoring_rubric>" in prefix


def test_build_user_message_includes_all_papers_as_json():
    msg = build_user_message([_paper("2604.00001", hf_upvotes=5), _paper("2604.00002")])
    assert "2604.00001" in msg and "2604.00002" in msg
    assert '"hf_upvotes": 5' in msg


def test_score_papers_batches_of_batch_size(mocker):
    mock = mocker.patch("recommender.evaluate.complete_json_array")
    mock.side_effect = [
        [{"id": "2604.00001", "score": 7.0,
          "breakdown": {"relevance": 7, "quality": 7, "field_importance": 7},
          "why": "reason"}],
        [{"id": "2604.00002", "score": 3.0,
          "breakdown": {"relevance": 3, "quality": 3, "field_importance": 3},
          "why": "reason"}],
    ]
    papers = [_paper("2604.00001"), _paper("2604.00002")]
    scores = score_papers(
        papers,
        primary="P",
        secondary="S",
        run_id=1,
        model="openrouter/anthropic/claude-haiku-4-5",
        batch_size=1,  # force two calls
    )
    assert mock.call_count == 2
    assert {s.arxiv_id for s in scores} == {"2604.00001", "2604.00002"}
    s1 = next(s for s in scores if s.arxiv_id == "2604.00001")
    assert s1.score == 7.0
    assert s1.breakdown["quality"] == 7
    assert s1.why == "reason"


def test_score_papers_skips_papers_whose_batch_fails(mocker):
    mock = mocker.patch("recommender.evaluate.complete_json_array")
    mock.side_effect = [ValueError("parse fail"), [
        {"id": "2604.00002", "score": 3.0,
         "breakdown": {"relevance": 3, "quality": 3, "field_importance": 3},
         "why": "r"},
    ]]
    papers = [_paper("2604.00001"), _paper("2604.00002")]
    scores = score_papers(
        papers, primary="P", secondary="", run_id=1,
        model="anthropic/claude-haiku-4-5", batch_size=1,
    )
    assert {s.arxiv_id for s in scores} == {"2604.00002"}
