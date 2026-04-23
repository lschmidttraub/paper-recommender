import json
from datetime import datetime, timezone

from recommender.sources.huggingface import parse_daily


def test_parse_daily_normalizes_papers_with_upvotes(fixtures_dir):
    raw = json.loads((fixtures_dir / "hf_daily_sample.json").read_text())
    papers = parse_daily(raw, now=datetime(2026, 4, 24, tzinfo=timezone.utc))
    assert len(papers) == 2
    p0 = papers[0]
    assert p0.arxiv_id == "2604.00001"
    assert p0.hf_upvotes == 42
    assert p0.sources == ("hf",)
    assert p0.authors == ("Alice Author", "Bob Author")
    assert p0.url == "https://arxiv.org/abs/2604.00001"
