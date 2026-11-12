CREATE TABLE IF NOT EXISTS guesses (
  id          TEXT PRIMARY KEY,
  created_at  INTEGER NOT NULL,          -- unix seconds
  ip_hash     TEXT NOT NULL,             -- salted hash, for rate limiting only
  place       TEXT NOT NULL,
  country     TEXT NOT NULL,
  model       TEXT NOT NULL,
  guess_lat   REAL,
  guess_lon   REAL,
  confidence  TEXT,                      -- very low | low | medium | high
  chance_pct  INTEGER,                   -- Claude's stated chance of being within 25 km
  reasoning   TEXT,
  candidates  TEXT,                      -- JSON list of OpenStreetMap matches shown to the visitor
  true_lat    REAL,
  true_lon    REAL,
  true_name   TEXT,
  km          REAL,
  status      TEXT NOT NULL DEFAULT 'pending'   -- pending | confirmed | rejected
);
CREATE INDEX IF NOT EXISTS guesses_ip_time ON guesses (ip_hash, created_at);
CREATE INDEX IF NOT EXISTS guesses_status_time ON guesses (status, created_at);
