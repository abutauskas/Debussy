# Running debussylabs.com: sign-in, API keys and the hosted API

```
browser ──► Cloudflare Pages ─┬─ site/        static pages (home, news, docs, sign-in, dashboard)
                              ├─ functions/   /api/* (accounts, keys) and /v1/* (the API gateway)
                              └─ D1 database  accounts, key fingerprints, usage counts
                                     │
           /v1/* requests ──────────►└──► model server: python -m debussy.server  (a machine you rent)
```

- **Accounts:** "Sign in with GitHub". No passwords. We keep GitHub id, username, name and primary verified email.
- **Waitlist:** new accounts start on the waitlist, and you approve them (or set `AUTO_APPROVE=true`).
- **Keys:** approved users create keys in the dashboard. A key (`dbsy_` + 40 random characters) is shown **once**;
  the database keeps only its SHA-256 fingerprint. Users can revoke keys and delete their account themselves.
- **Gateway:** `/v1/*` checks the key, applies per-key limits (default 20/minute, 500/day), counts usage (never the
  content), and forwards the request to the model server with a shared gateway token.

Everything except the model server runs on Cloudflare. Until the model server is connected, `/v1/*` answers
"not live yet" (503), and sign-in, the waitlist and keys all work.

## 1. Database (D1)
1. Cloudflare dashboard → **Storage & Databases → D1 → Create** a database named `debussy`.
2. In its **Console**, paste `cloudflare/schema.sql` and run it.
3. Pages project → **Settings → Bindings → Add → D1 database**: variable name `DB`, database `debussy`. Redeploy.

## 2. GitHub sign-in
1. GitHub → **Settings → Developer settings → OAuth Apps → New OAuth App**
   - Homepage URL: `https://debussylabs.com`
   - Authorization callback URL: `https://debussylabs.com/api/auth/callback`
2. Copy the **Client ID**, then **Generate a new client secret**.
3. Pages project → **Settings → Variables and secrets**:
   - `GITHUB_CLIENT_ID`: plain text
   - `GITHUB_CLIENT_SECRET`: type **Secret** (never commit it or paste it anywhere else)
4. Redeploy, then try **Sign in** on the site. You'll land on the waitlist screen.

Sign-in always runs on `https://debussylabs.com` (visitors on `www` or the `*.pages.dev` address are sent there
first), because GitHub only returns to the registered callback URL. If the main address ever changes, set `SITE_URL`.

Optional variables: `AUTO_APPROVE` (`true` skips the waitlist), `RATE_PER_MINUTE` (20), `RATE_PER_DAY` (500),
`MAX_KEYS` per user (5).

## 3. Approving people
In the D1 console:
```sql
SELECT id, login, email, datetime(created_at, 'unixepoch') AS joined FROM users WHERE status = 'waitlist' ORDER BY created_at;
UPDATE users SET status = 'approved' WHERE login = 'their-github-username';
UPDATE users SET status = 'blocked'  WHERE login = 'someone';           -- blocks sign-in and the API
SELECT login, SUM(count) AS calls FROM usage JOIN api_keys k ON k.id = usage.key_id JOIN users u ON u.id = k.user_id
 WHERE bucket LIKE 'd:%' GROUP BY login ORDER BY calls DESC;            -- usage per user (daily counts)
```
There are no automatic emails yet; write to approved people yourself.

## 4. The model server
On the machine that runs Debussy (see the README for downloading the models):
```bash
export DEBUSSY_GATEWAY_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")   # keep this
python -m debussy.server --variant duo --host 127.0.0.1 --port 8000
```
Expose it over HTTPS without opening ports, for example with a Cloudflare Tunnel (`cloudflared`) on a hostname such as
`model.debussylabs.com`. Then set in the Pages project:
- `INFERENCE_URL`: plain text, e.g. `https://model.debussylabs.com`
- `INFERENCE_TOKEN`: type **Secret**, the same value as `DEBUSSY_GATEWAY_TOKEN`

The server answers only requests carrying that token, handles one decision at a time, and logs only method, path
and status. It also needs `llama-server` running with K2-Horizon for the Duo and K2 variants (`docs/K2_EXPERIMENT.md`).

**Where to run it:** a CPU server is cheap, but at 6–13 s per decision it suits only a small beta. A GPU server
should bring decisions well under a second (not yet measured), but it costs money every hour it's on.
Serverless GPUs bill per second but add start-up delays. Only open the waitlist as far as your hosting budget allows.

## 5. Before you open sign-ups
- Set up `privacy@debussylabs.com` (Cloudflare **Email Routing** can forward it to your inbox for free). The privacy
  notice (`site/privacy/`) gives that address.
- Read `site/privacy/index.html` and make sure it matches how you actually run things. If you serve people in the EU,
  the GDPR also expects the operator's name and contact details; have it checked by someone qualified.
- Make the GitHub repository public, so the site's GitHub and model-card links work.

## Local testing
`cloudflare/test/README.md` runs the whole flow on your machine (wrangler, a local D1, a fake GitHub and the API server
in `--fake` mode). The site's header and footer are shared: edit `scripts/site/header.html` or `footer.html`, then
run `python scripts/site/stamp.py`.
