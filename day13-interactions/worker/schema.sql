CREATE TABLE IF NOT EXISTS drawings (
  id          TEXT PRIMARY KEY,
  created_at  INTEGER NOT NULL,          -- unix seconds
  ip_hash     TEXT NOT NULL,             -- salted daily hash, for rate limiting only
  country     TEXT NOT NULL,             -- Natural Earth ADM0_A3, e.g. UKR
  ring        TEXT NOT NULL,             -- JSON [[lon, lat], ...], 2 decimals
  n_points    INTEGER NOT NULL,
  seq         INTEGER NOT NULL,          -- position in the visitor's round (1 = first drawing)
  session     TEXT,                      -- random id per round, so drawings from one round can be grouped
  hidden      INTEGER NOT NULL DEFAULT 0 -- 1 = removed by hand (spam); never shown or counted
);
CREATE INDEX IF NOT EXISTS drawings_ip_time ON drawings (ip_hash, created_at);
CREATE INDEX IF NOT EXISTS drawings_country ON drawings (country, hidden, created_at);
