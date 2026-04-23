from __future__ import annotations

import json
import logging
import re
from typing import Any

import litellm
from litellm import completion as litellm_completion  # re-exported so tests can monkeypatch

log = logging.getLogger(__name__)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("anthropic/") or model.startswith("openrouter/anthropic/")


def _prepend_cacheable(messages: list[dict], cacheable_prefix: str, model: str) -> list[dict]:
    if not cacheable_prefix:
        return messages
    if _is_anthropic_model(model):
        # Anthropic content-block form with cache_control marker
        first_user_idx = next(i for i, m in enumerate(messages) if m["role"] == "user")
        original = messages[first_user_idx]["content"]
        original_text = original if isinstance(original, str) else original[0].get("text", "")
        new_content = [
            {
                "type": "text",
                "text": cacheable_prefix,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": original_text},
        ]
        out = list(messages)
        out[first_user_idx] = {"role": "user", "content": new_content}
        return out
    # Non-anthropic: plain string concatenation, relying on automatic prefix caching.
    first_user_idx = next(i for i, m in enumerate(messages) if m["role"] == "user")
    original = messages[first_user_idx]["content"]
    if not isinstance(original, str):
        original = "".join(p.get("text", "") for p in original if isinstance(p, dict))
    out = list(messages)
    out[first_user_idx] = {"role": "user", "content": f"{cacheable_prefix}\n\n{original}"}
    return out


def _call_llm(model: str, messages: list[dict], **kwargs) -> str:
    resp = litellm_completion(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content or ""


def _extract_json(raw: str, pattern: re.Pattern[str]) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = pattern.search(raw)
    if not m:
        raise ValueError(f"No JSON match in LLM output: {raw[:200]!r}")
    return json.loads(m.group(0))


def complete_json_array(
    *,
    model: str,
    messages: list[dict],
    cacheable_prefix: str = "",
    max_retries: int = 1,
    **kwargs,
) -> list[dict]:
    msgs = _prepend_cacheable(messages, cacheable_prefix, model)
    last_err: Exception | None = None
    for _ in range(max_retries + 1):
        raw = _call_llm(model, msgs, **kwargs)
        try:
            parsed = _extract_json(raw, _JSON_ARRAY_RE)
            if isinstance(parsed, list):
                return parsed
            raise ValueError("Expected JSON array, got object")
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            log.warning("LLM JSON array parse failed: %s", e)
    raise ValueError(f"LLM never returned parseable JSON array: {last_err}")


def complete_json_object(
    *,
    model: str,
    messages: list[dict],
    cacheable_prefix: str = "",
    max_retries: int = 1,
    **kwargs,
) -> dict:
    msgs = _prepend_cacheable(messages, cacheable_prefix, model)
    last_err: Exception | None = None
    for _ in range(max_retries + 1):
        raw = _call_llm(model, msgs, **kwargs)
        try:
            parsed = _extract_json(raw, _JSON_OBJECT_RE)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Expected JSON object, got array")
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            log.warning("LLM JSON object parse failed: %s", e)
    raise ValueError(f"LLM never returned parseable JSON object: {last_err}")
