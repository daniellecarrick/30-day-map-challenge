# Day 13 drawing store (Cloudflare Worker)

Saves visitors' country outlines to a D1 database and serves them back for the heat map.
No API keys and no running cost beyond Cloudflare's free tier.

## Deploy (one time)
Run these from this folder. `npx` downloads wrangler on first use.

1. `npx wrangler login` (skip if you're still logged in from day 10)
2. `npx wrangler d1 create day13-drawings`, then paste the `database_id` it prints into `wrangler.toml`.
3. `npx wrangler d1 execute day13-drawings --remote --file=schema.sql`
4. `npx wrangler deploy` prints the Worker URL (https://day13-draw-the-countries.<you>.workers.dev).
5. Rebuild the page with that URL: `cd .. && WORKER_URL=<url> ../.venv/bin/python build.py`

## Knobs (wrangler.toml)
`HOURLY_LIMIT` drawings per visitor per hour (60), `DAILY_LIMIT` for everyone (20,000),
`ALLOWED_ORIGINS` (the Pages site plus localhost:8000 for testing).

## Look at the data
`npx wrangler d1 execute day13-drawings --remote --command "SELECT country, COUNT(*) FROM drawings WHERE hidden = 0 GROUP BY country ORDER BY 2 DESC"`

## Remove a junk drawing
Drawings are shapes only, so there's little room for abuse. To hide one (it stays in the table but is
never shown or counted): find its id in `GET /drawings?country=XXX`, then
`npx wrangler d1 execute day13-drawings --remote --command "UPDATE drawings SET hidden = 1 WHERE id = '<id>'"`

## Make the PNG from the latest drawings
`cd .. && WORKER_URL=<url> ../.venv/bin/python build.py` fetches every country's drawings into
`data/drawings/`, prints the stats, writes the headline from them, and redraws `out/map.png`.
Until a country has 10 drawings its panel says "not enough to map".

## Local testing
`npx wrangler d1 execute day13-drawings --local --file=schema.sql`, then `npx wrangler dev --local --port 8787`.
Put `ALLOWED_ORIGINS="http://localhost:<port>"` in `.dev.vars` (gitignored) for the port that serves the page.
