# Local test of sign-in, API keys and the gateway

Runs the real Pages Functions (`functions/`) and site (`site/`) in wrangler's local mode, with a local D1 database,
a stand-in for GitHub (`mock_github.py`) and the Debussy API server in `--fake` mode. Nothing here touches production.

```bash
# from the repository root, in a scratch folder
T=$(mktemp -d) && cp cloudflare/test/wrangler.toml cloudflare/test/mock_github.py $T/ && cd $T
ln -s "$OLDPWD/site" site && ln -s "$OLDPWD/functions" functions && ln -s "$OLDPWD/cloudflare" cloudflare
printf 'GITHUB_CLIENT_SECRET=test-secret\nINFERENCE_TOKEN=test-gw-token\n' > .dev.vars
npm i wrangler@4 && npx wrangler d1 execute debussy-local --local --file cloudflare/schema.sql
python3 mock_github.py &                                                     # fake GitHub on :8012
(cd "$OLDPWD" && DEBUSSY_GATEWAY_TOKEN=test-gw-token python3 -m debussy.server --fake --port 8011 &)
npx wrangler pages dev --port 8788 --ip 127.0.0.1                            # site + functions on :8788
```

Then sign in by visiting `/api/auth/github`, and replace github.com in the redirect with the callback
`/api/auth/callback?code=good-code&state=<state from the redirect>`. Approve the test user with
`npx wrangler d1 execute debussy-local --local --command "UPDATE users SET status='approved'"`.
