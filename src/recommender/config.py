from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    scoring_model: str
    bridging_model: str
    batch_size: int

    arxiv_categories: tuple[str, ...]
    arxiv_max_backfill_days: int

    on_interest_min: int
    on_interest_max: int
    on_interest_threshold: float
    hf_upvote_threshold_for_hot_surprise: int

    email_to: str
    email_from: str
    smtp_password: str

    db_path: Path
    digests_dir: Path
    logs_dir: Path
    memory_md: Path
    claude_projects_root: Path

    @classmethod
    def from_env(cls, project_root: Path, env_file: Path | None = None) -> "Settings":
        if env_file is None:
            env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        return cls(
            scoring_model=os.getenv("SCORING_MODEL", "openrouter/anthropic/claude-haiku-4-5"),
            bridging_model=os.getenv("BRIDGING_MODEL", "openrouter/anthropic/claude-haiku-4-5"),
            batch_size=_int("BATCH_SIZE", 20),
            arxiv_categories=tuple(
                os.getenv("ARXIV_CATEGORIES", "cs.LG,cs.AI,cs.CV,cs.CL,stat.ML").split(",")
            ),
            arxiv_max_backfill_days=_int("ARXIV_MAX_BACKFILL_DAYS", 7),
            on_interest_min=_int("ON_INTEREST_MIN", 5),
            on_interest_max=_int("ON_INTEREST_MAX", 15),
            on_interest_threshold=_float("ON_INTEREST_THRESHOLD", 7.0),
            hf_upvote_threshold_for_hot_surprise=_int("HF_UPVOTE_THRESHOLD", 10),
            email_to=_require("EMAIL_TO"),
            email_from=_require("EMAIL_FROM"),
            smtp_password=_require("GMAIL_APP_PASSWORD"),
            db_path=project_root / "data" / "papers.sqlite",
            digests_dir=project_root / "digests",
            logs_dir=project_root / "logs",
            memory_md=project_root / "MEMORY.md",
            claude_projects_root=Path.home() / ".claude" / "projects",
        )
