# Day 10 live-guess API (Cloudflare Worker)

Holds the Anthropic key, asks Claude for a guess (memory only), finds the real place on
OpenStreetMap, and saves confirmed guesses to a D1 database the map reads from.

## Deploy (one time)
Run these from this folder. `npx` downloads wrangler on first use.

1. `npx wrangler login`
2. `npx wrangler d1 create day10-guesses`, then paste the `database_id` it prints into `wrangler.toml`.
3. `npx wrangler d1 execute day10-guesses --remote --file=schema.sql`
4. `npx wrangler secret put ANTHROPIC_API_KEY` (paste the key at the prompt; it is never stored in a file)
5. `npx wrangler deploy` prints the Worker URL (https://day10-claude-geo-guess.<you>.workers.dev).
6. Rebuild the page with that URL: `WORKER_URL=<url> ../../.venv/bin/python ../build.py`

Also set a monthly spend limit on the key at console.anthropic.com.

## Knobs (wrangler.toml)
`MODEL` (default claude-haiku-5-5), `HOURLY_LIMIT` per visitor, `DAILY_LIMIT` for everyone,
`ALLOWED_ORIGINS` (add http://localhost:8000 to test locally).

## Look at the data
`npx wrangler d1 execute day10-guesses --remote --command "SELECT place,country,km,confidence FROM guesses WHERE status='confirmed' ORDER BY km DESC"`
