from datetime import datetime, timezone

import pytest
import requests

from recommender.sources.arxiv import build_query, fetch, parse_atom


class _FakeResp:
    def __init__(self, status_code, *, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Error", response=self)


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # (url, headers) per request

    def get(self, url, timeout=None, headers=None):
        self.calls.append((url, headers or {}))
        return self._responses.pop(0)


def test_fetch_retries_on_429_then_succeeds(fixtures_dir):
    feed = (fixtures_dir / "arxiv_sample.atom").read_text()
    sess = _FakeSession([
        _FakeResp(429, headers={"Retry-After": "2"}),
        _FakeResp(200, text=feed),
    ])
    slept: list[float] = []
    papers = fetch(
        ("cs.LG",),
        since=datetime(2026, 6, 1, tzinfo=timezone.utc),
        session=sess,
        sleep=slept.append,
    )
    assert len(papers) == 2
    assert slept == [2.0]  # honored the Retry-After header
    assert len(sess.calls) == 2
    assert "paper-recommender" in sess.calls[0][1].get("User-Agent", "")


def test_fetch_raises_after_exhausting_retries():
    sess = _FakeSession([_FakeResp(429) for _ in range(10)])
    with pytest.raises(requests.exceptions.HTTPError):
        fetch(
            ("cs.LG",),
            since=datetime(2026, 6, 1, tzinfo=timezone.utc),
            session=sess,
            max_retries=2,
            sleep=lambda _s: None,
        )
    assert len(sess.calls) == 3  # initial attempt + 2 retries


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
