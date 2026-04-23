# paper-recommender

Single-user daily ML paper digest. Scrapes arXiv + Hugging Face Daily Papers,
scores against a hand-curated `MEMORY.md`, and emails a markdown digest.

## Setup

1. Install `uv` if needed: https://docs.astral.sh/uv/
2. `uv sync --extra dev`
3. Copy `.env.example` to `.env` and fill in:
   - At least one of `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`
   - `SCORING_MODEL` and `BRIDGING_MODEL` (LiteLLM format, e.g. `openrouter/anthropic/claude-haiku-4-5`)
   - `EMAIL_TO`, `EMAIL_FROM`
   - `GMAIL_APP_PASSWORD` — see "Gmail app password" below
4. Edit `MEMORY.md` to describe your research interests.

### Gmail app password

1. Enable 2FA on the Gmail account.
2. Visit https://myaccount.google.com/apppasswords
3. Create a password for "Mail / Other: paper-recommender". Paste into `.env` as `GMAIL_APP_PASSWORD`.

## Usage

```bash
# Manual run
uv run python -m recommender

# Dry run: writes the markdown digest, skips email and digest_entries
uv run python -m recommender --dry-run

# Regenerate a specific day (overwrites digests/YYYY-MM-DD.md, remails)
uv run python -m recommender --force-date 2026-04-24

# Look back N days instead of since-last-run
uv run python -m recommender --backfill 3
```

### Cron

Add to your crontab (`crontab -e`):

```cron
0 7 * * * cd /home/YOU/src/paper-recommender && /usr/bin/env -S uv run python -m recommender >> logs/cron.log 2>&1
```

## Tests

```bash
uv run pytest
```
