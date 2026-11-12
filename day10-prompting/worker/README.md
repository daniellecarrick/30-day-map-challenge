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

## API key expires Jan 31, 2027
**What happens if nobody renews it:** Anthropic starts rejecting every guess (a 401). Visitors see
"Guessing is paused while this project's API key is renewed. The map and earlier guesses still work."
Nothing else breaks, and nothing is lost.

**Renew (2 minutes; do it the week before, ideally by Jan 24):**
1. console.anthropic.com → API keys → create a new key (name it `day10 2027`). Give it the same monthly spend limit as the old one.
2. From this folder, run `npx wrangler secret put ANTHROPIC_API_KEY` and paste the new key at the prompt. No redeploy is needed.
3. Check it:
   `curl -X POST https://day10-claude-geo-guess.daniellecarrick.workers.dev/guess -H 'Content-Type: application/json' -d '{"place":"Lefors","country":"USA"}'`
   You should get a guess back. (This saves one unconfirmed row, which never shows on the map.)
4. Delete the old key in the console.

**If you see it fail:** `npx wrangler tail` prints the real reason from Anthropic (`anthropic 401` for a bad or expired key).

## Other errors visitors can see
| Situation | What the page says | Fix |
|---|---|---|
| A visitor makes more than 8 guesses in an hour | "To prevent abuse, each user is limited to 8 guesses a session. You can guess again in N minutes." The allowance is a rolling hour, so it resets by itself. | None. Change `HOURLY_LIMIT` to adjust. |
| Everyone together passes 300 guesses in 24 hours | "Claude has reached today's limit of guesses for everyone. Come back tomorrow." | Raise `DAILY_LIMIT` if you want more. |
| API credits or the monthly spend limit are used up | "This project's budget for Claude guesses has run out, so new guesses are paused." | Add credits or raise the limit in the console. No redeploy needed. |
| Key expired or revoked | "Guessing is paused while this project's API key is renewed." | See the section above. |
| Claude is overloaded | "Claude is busy right now. Try again in a minute." | None. |
