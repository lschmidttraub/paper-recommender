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

## Scheduling

### GitHub Actions (recommended)

`.github/workflows/digest.yml` runs the pipeline daily in the cloud, so it does
not depend on any laptop being awake. Local `cron` silently skips the job
whenever the machine is suspended or off at the scheduled minute (cronie does
not run missed jobs on resume), which is the usual reason digests stop arriving.

One-time setup — add the secrets the workflow reads (run from the repo, or set
them in the GitHub UI under Settings → Secrets and variables → Actions):

```bash
gh secret set OPENROUTER_API_KEY   # or whichever provider key SCORING_MODEL needs
gh secret set GMAIL_APP_PASSWORD
gh secret set EMAIL_TO
gh secret set EMAIL_FROM
gh secret set ZOTERO_API_KEY       # optional (primary interest signal)
gh secret set ZOTERO_USER_ID       # optional
```

Then trigger a test run from the Actions tab ("Run workflow") or with
`gh workflow run "Daily digest"`. Notes:

- The schedule is `0 5 * * *` UTC = 07:00 Berlin in summer (06:00 in winter; GitHub cron has no DST).
- The SQLite DB (scoring cache + dedup state) is persisted via the Actions cache, not committed to the repo.
- The `~/.claude/projects` secondary interest signal is local-only and absent on the runner; `MEMORY.md` + Zotero remain the primary signal.
- GitHub disables scheduled workflows after 60 days with no repo activity — push a commit or re-enable if that happens.

### Local cron (laptop-dependent)

Runs only when the machine is awake at the scheduled minute. Add to your crontab (`crontab -e`):

```cron
0 7 * * * cd /home/YOU/src/paper-recommender && /usr/bin/env -S uv run python -m recommender >> logs/cron.log 2>&1
```

For a laptop, a `systemd` timer with `Persistent=true` is more reliable than cron
because it runs the missed job on resume instead of skipping it.

## Tests

```bash
uv run pytest
```
