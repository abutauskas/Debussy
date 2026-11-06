-- Debussy Labs accounts and API keys (Cloudflare D1, SQLite).
-- Apply once:  npx wrangler d1 execute debussy --remote --file cloudflare/schema.sql
-- Stored per user: GitHub id and username, display name, primary verified email. No passwords, no GitHub tokens.
-- API keys and session tokens are stored only as SHA-256 hashes. Usage is counted per key; request contents are never stored.

CREATE TABLE IF NOT EXISTS users (
  id          INTEGER PRIMARY KEY,
  github_id   INTEGER NOT NULL UNIQUE,
  login       TEXT    NOT NULL,
  name        TEXT,
  email       TEXT,
  status      TEXT    NOT NULL DEFAULT 'waitlist' CHECK (status IN ('waitlist', 'approved', 'blocked')),
  created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  token_hash  TEXT    PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS api_keys (
  id           INTEGER PRIMARY KEY,
  user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name         TEXT    NOT NULL,
  prefix       TEXT    NOT NULL,          -- first characters, shown in the dashboard to tell keys apart
  key_hash     TEXT    NOT NULL UNIQUE,   -- SHA-256 of the full key; the key itself is shown once and never stored
  created_at   INTEGER NOT NULL,
  last_used_at INTEGER,
  revoked_at   INTEGER
);
CREATE INDEX IF NOT EXISTS api_keys_user ON api_keys(user_id);

CREATE TABLE IF NOT EXISTS usage (
  key_id  INTEGER NOT NULL,
  bucket  TEXT    NOT NULL,               -- 'm:<unix minute>' or 'd:<YYYY-MM-DD>'
  count   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (key_id, bucket)
);
