from __future__ import annotations

import argparse
import logging
import sys
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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_env(project_root=Path.cwd())
    run_pipeline(
        settings,
        force_date=args.force_date,
        dry_run=args.dry_run,
        no_email=args.no_email,
        backfill_days=args.backfill,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
