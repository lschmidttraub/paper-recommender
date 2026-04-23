from datetime import datetime, timezone

import pytest

from recommender.models import Paper, Score
from recommender.store import Store
from recommender.surprise import pick_bridging, pick_hot_outside_field


def _paper(arxiv_id, *, sources=("arxiv",), hf_upvotes=None):
    return Paper(
        arxiv_id=arxiv_id,
        title=f"T {arxiv_id}",
        abstract="abs",
        authors=("A",),
        categories=("cs.LG",),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        sources=sources,
        hf_upvotes=hf_upvotes,
        first_seen_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


def _score(arxiv_id, run_id, value):
    return Score(
        arxiv_id=arxiv_id, run_id=run_id, model="m", score=value,
        breakdown={"relevance": 0, "quality": 0, "field_importance": 0},
        why="", scored_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


@pytest.fixture
def populated_store(tmp_path):
    store = Store(tmp_path / "t.sqlite")
    store.init_db()
    store.upsert_papers([
        _paper("2604.00001", sources=("arxiv", "hf"), hf_upvotes=50),
        _paper("2604.00002", sources=("arxiv", "hf"), hf_upvotes=25),
        _paper("2604.00003", sources=("arxiv",)),  # not on HF
        _paper("2604.00004", sources=("arxiv", "hf"), hf_upvotes=5),  # under threshold
    ])
    run_id = store.start_run()
    store.save_scores([
        _score("2604.00001", run_id, 3.0),  # low score, hot HF => eligible
        _score("2604.00002", run_id, 4.5),  # lower upvotes but eligible
        _score("2604.00003", run_id, 2.0),  # not on HF => ineligible
        _score("2604.00004", run_id, 2.0),  # HF upvotes under threshold => ineligible
    ])
    return store, run_id


def test_pick_hot_outside_field_returns_top_upvoted_low_score(populated_store):
    store, run_id = populated_store
    pick = pick_hot_outside_field(store, run_id=run_id, upvote_threshold=10)
    assert pick is not None
    assert pick.arxiv_id == "2604.00001"


def test_pick_hot_outside_field_returns_none_when_no_candidates(populated_store):
    store, run_id = populated_store
    pick = pick_hot_outside_field(store, run_id=run_id, upvote_threshold=1000)
    assert pick is None


def test_pick_bridging_returns_none_when_too_few_candidates(populated_store, mocker):
    store, run_id = populated_store
    mock = mocker.patch("recommender.surprise.complete_json_object")
    # Only 2 papers in 3-6 band (0001 at 3.0, 0002 at 4.5); below min_candidates default 10.
    pick = pick_bridging(
        store, run_id=run_id, primary="interests",
        model="anthropic/claude-haiku-4-5", min_candidates=10,
    )
    assert pick is None
    mock.assert_not_called()


def test_pick_bridging_calls_llm_and_returns_chosen(populated_store, mocker):
    store, run_id = populated_store
    # Add more mid-band papers to clear the min_candidates bar
    extra = [_paper(f"2604.{i:05d}") for i in range(10, 30)]
    store.upsert_papers(extra)
    store.save_scores([_score(p.arxiv_id, run_id, 4.0) for p in extra])

    mock = mocker.patch("recommender.surprise.complete_json_object")
    mock.return_value = {"id": "2604.00010", "bridge_reason": "methodological parallel"}
    pick = pick_bridging(
        store, run_id=run_id, primary="interests",
        model="anthropic/claude-haiku-4-5", min_candidates=10,
    )
    assert pick is not None
    paper, reason = pick
    assert paper.arxiv_id == "2604.00010"
    assert reason == "methodological parallel"
    mock.assert_called_once()
