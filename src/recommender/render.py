from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from recommender.models import Paper

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"


@dataclass(frozen=True)
class OnInterestItem:
    paper: Paper
    score: float
    why: str
    breakdown: dict[str, int]


@dataclass(frozen=True)
class HotOutsideItem:
    paper: Paper
    score: float


@dataclass(frozen=True)
class BridgingItem:
    paper: Paper
    bridge_reason: str


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_digest(
    *,
    date: str,
    on_interest: list[OnInterestItem],
    hot_outside: HotOutsideItem | None,
    bridging: BridgingItem | None,
    total_seen: int,
    scoring_model: str,
    run_duration_s: float,
    errors: str,
) -> str:
    surprise_count = int(hot_outside is not None) + int(bridging is not None)
    tmpl = _env().get_template("digest.md.j2")
    return tmpl.render(
        date=date,
        on_interest=on_interest,
        hot_outside=hot_outside,
        bridging=bridging,
        total_seen=total_seen,
        scoring_model=scoring_model,
        surprise_count=surprise_count,
        run_duration_s=f"{run_duration_s:.1f}",
        errors=errors,
    )
