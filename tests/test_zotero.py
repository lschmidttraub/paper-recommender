import json
from unittest.mock import MagicMock

from recommender.sources.zotero import ZoteroItem, fetch_items, format_library


def test_fetch_items_parses_response(fixtures_dir, mocker):
    raw = json.loads((fixtures_dir / "zotero_sample.json").read_text())
    session = MagicMock()
    response = MagicMock()
    response.json.side_effect = [raw, []]  # second call (next page) returns empty
    response.raise_for_status.return_value = None
    session.get.return_value = response

    items = fetch_items(api_key="k", user_id="123", session=session, max_items=500)

    assert len(items) == 2
    sae = items[0]
    assert sae.title.startswith("Sparse Autoencoders")
    assert sae.creators == ("Hoagy Cunningham", "Aidan Ewart")
    assert sae.tags == ("interpretability", "SAE")
    assert sae.year == "2023"
    assert sae.item_type == "journalArticle"


def test_fetch_items_skips_items_without_title(mocker):
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = [
        {"key": "X", "data": {"key": "X", "itemType": "note"}},  # no title
        {"key": "Y", "data": {"key": "Y", "itemType": "journalArticle", "title": "Real paper"}},
    ]
    response.raise_for_status.return_value = None
    session.get.return_value = response
    items = fetch_items(api_key="k", user_id="123", session=session, max_items=500)
    assert [i.title for i in items] == ["Real paper"]


def test_fetch_items_respects_max_items(mocker):
    """If max_items < page size, only request that many."""
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = [
        {"key": str(i), "data": {"key": str(i), "itemType": "journalArticle", "title": f"T{i}"}}
        for i in range(50)
    ]
    response.raise_for_status.return_value = None
    session.get.return_value = response
    items = fetch_items(api_key="k", user_id="123", session=session, max_items=10)
    assert len(items) == 10
    # Verify we asked for limit=10, not 100
    _args, kwargs = session.get.call_args
    assert kwargs["params"]["limit"] == 10


def test_format_library_empty():
    assert format_library([]) == ""


def test_format_library_includes_title_year_creators_tags():
    items = [
        ZoteroItem(
            key="K",
            title="Some Paper",
            creators=("Alice Smith", "Bob Jones"),
            year="2024",
            tags=("topic-a", "topic-b"),
            item_type="journalArticle",
            date_added="2024-01-01T00:00:00Z",
        ),
    ]
    out = format_library(items)
    assert "Some Paper" in out
    assert "2024" in out
    assert "Alice Smith, Bob Jones" in out
    assert "topic-a, topic-b" in out


def test_format_library_truncates_authors_after_5():
    items = [
        ZoteroItem(
            key="K", title="Big Author Paper",
            creators=tuple(f"A{i}" for i in range(10)),
            year="", tags=(), item_type="journalArticle", date_added="",
        )
    ]
    out = format_library(items)
    assert "et al." in out
    assert "A0, A1, A2, A3, A4 et al." in out
