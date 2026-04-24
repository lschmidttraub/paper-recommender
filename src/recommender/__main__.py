from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from recommender.config import Settings
from recommender.main import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recommender")
    parser.add_argument("--dry-run", action="store_true", help="Write markdown but skip email and digest_entries persistence")
    parser.add_argument("--no-email", action="store_true", help="Persist digest but skip email")
    parser.add_argument("--backfill", type=int, default=None, help="Look back N days instead of since-last-run")
    parser.add_argument("--force-date", type=str, default=None, help="Regenerate the digest for YYYY-MM-DD")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--email-only", action="store_true",
                        help="Send existing digest by email without scraping/scoring (requires --force-date or defaults to today)")
    args = parser.parse_args(argv)

    # File handler: INFO+ to logs/YYYY-MM-DD.log
    # Stderr handler: WARNING+ (cron captures this for failure alerts)
    root = logging.getLogger()
    root.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    today = datetime.now(timezone.utc).date().isoformat()
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / f"{today}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)

    root.addHandler(file_handler)
    root.addHandler(stderr_handler)
    settings = Settings.from_env(project_root=Path.cwd())
    run_pipeline(
        settings,
        force_date=args.force_date,
        dry_run=args.dry_run,
        no_email=args.no_email,
        email_only=args.email_only,
        backfill_days=args.backfill,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
