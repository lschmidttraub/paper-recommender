import json
from unittest.mock import MagicMock

import pytest

from recommender.llm import complete_json_array, complete_json_object


@pytest.fixture
def mock_litellm(mocker):
    m = mocker.patch("recommender.llm.litellm_completion")
    # default: return a valid JSON array
    m.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='[{"id": "1", "score": 7}]'))])
    return m


def test_complete_json_array_parses_strict(mock_litellm):
    out = complete_json_array(model="anthropic/claude-haiku-4-5", messages=[{"role": "user", "content": "x"}])
    assert out == [{"id": "1", "score": 7}]


def test_complete_json_array_extracts_array_from_noisy_output(mock_litellm):
    mock_litellm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='Here you go: [{"id": "1"}]  thanks!'))]
    )
    out = complete_json_array(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
    assert out == [{"id": "1"}]


def test_complete_json_array_retries_on_parse_failure(mock_litellm):
    mock_litellm.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="not json"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content='[{"id": "2"}]'))]),
    ]
    out = complete_json_array(model="anthropic/claude-haiku-4-5", messages=[{"role": "user", "content": "x"}])
    assert out == [{"id": "2"}]
    assert mock_litellm.call_count == 2


def test_complete_json_array_raises_after_final_failure(mock_litellm):
    mock_litellm.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="still not json"))])
    with pytest.raises(ValueError):
        complete_json_array(model="anthropic/claude-haiku-4-5", messages=[{"role": "user", "content": "x"}])


def test_cache_control_applied_for_anthropic_models(mock_litellm):
    cacheable = "LONG CACHEABLE BLOCK"
    complete_json_array(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "x"}],
        cacheable_prefix=cacheable,
    )
    args, kwargs = mock_litellm.call_args
    msgs = kwargs.get("messages") or args[1]
    # First user message should include the cacheable block with cache_control
    parts = msgs[0]["content"]
    assert isinstance(parts, list)
    has_cache_marker = any(
        p.get("cache_control", {}).get("type") == "ephemeral"
        for p in parts if isinstance(p, dict)
    )
    assert has_cache_marker
    assert any(cacheable in (p.get("text") or "") for p in parts if isinstance(p, dict))


def test_cache_control_not_applied_for_openai(mock_litellm):
    complete_json_array(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
        cacheable_prefix="LONG",
    )
    args, kwargs = mock_litellm.call_args
    msgs = kwargs.get("messages") or args[1]
    # For non-anthropic, we pass a plain string prefix (no cache_control marker)
    content = msgs[0]["content"]
    assert isinstance(content, str)
    assert "LONG" in content


def test_complete_json_object_parses_object(mock_litellm):
    mock_litellm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"id": "1", "bridge_reason": "because"}'))]
    )
    out = complete_json_object(model="anthropic/claude-haiku-4-5", messages=[{"role": "user", "content": "x"}])
    assert out == {"id": "1", "bridge_reason": "because"}
