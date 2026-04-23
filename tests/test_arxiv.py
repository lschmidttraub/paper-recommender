from datetime import datetime, timezone

from recommender.sources.arxiv import build_query, parse_atom


def test_parse_atom_normalizes_papers(fixtures_dir):
    feed_text = (fixtures_dir / "arxiv_sample.atom").read_text()
    papers = parse_atom(feed_text, now=datetime(2026, 4, 24, tzinfo=timezone.utc))
    assert len(papers) == 2
    p0 = papers[0]
    assert p0.arxiv_id == "2604.00001"   # version suffix stripped
    assert p0.title == "Sparse Autoencoders Find Interpretable Features"
    assert p0.authors == ("Alice Author", "Bob Author")
    assert "cs.LG" in p0.categories and "cs.AI" in p0.categories
    assert p0.sources == ("arxiv",)
    assert p0.hf_upvotes is None


def test_build_query_includes_all_categories_and_date_window():
    q = build_query(
        categories=("cs.LG", "cs.AI"),
        since=datetime(2026, 4, 20, tzinfo=timezone.utc),
        until=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )
    assert "cs.LG" in q and "cs.AI" in q
    assert "202604200000" in q.replace("-", "").replace(":", "").replace(" ", "")
    assert "202604240000" in q.replace("-", "").replace(":", "").replace(" ", "")
