// GET /api/auth/callback: GitHub sends the user back here. We check the state, read the GitHub profile once
// (the GitHub token is not stored), create or update the account, and start a session.
import { OAUTH_COOKIE, SESSION_COOKIE, SESSION_DAYS, getCookie, now, randomBase62, redirect, setCookie, sha256hex } from "../../../cloudflare/lib.js";

const fail = (code) => redirect(`/login/?error=${code}`, [setCookie(OAUTH_COOKIE, "", 0)]);

export async function onRequestGet({ request, env }) {
  if (!env.GITHUB_CLIENT_ID || !env.GITHUB_CLIENT_SECRET || !env.DB) return fail("unavailable");
  const url = new URL(request.url);
  const code = url.searchParams.get("code"), state = url.searchParams.get("state"), saved = getCookie(request, OAUTH_COOKIE);
  if (!code || !state || !saved || state !== saved) return fail("state");

  // GITHUB_TOKEN_URL / GITHUB_API_URL point at a fake GitHub for local tests; they are ignored on any real host.
  const local = ["127.0.0.1", "localhost"].includes(url.hostname);
  const tokenRes = await fetch((local && env.GITHUB_TOKEN_URL) || "https://github.com/login/oauth/access_token", {
    method: "POST", headers: { accept: "application/json", "content-type": "application/json" },
    body: JSON.stringify({ client_id: env.GITHUB_CLIENT_ID, client_secret: env.GITHUB_CLIENT_SECRET, code,
                           redirect_uri: new URL("/api/auth/callback", request.url).toString() }),
  });
  const token = tokenRes.ok ? (await tokenRes.json()).access_token : null;
  if (!token) return fail("github");
  const api = (local && env.GITHUB_API_URL) || "https://api.github.com";
  const gh = async (path) => {
    const r = await fetch(api + path, { headers: { authorization: `Bearer ${token}`, accept: "application/vnd.github+json", "user-agent": "debussylabs.com" } });
    return r.ok ? r.json() : null;
  };
  const profile = await gh("/user");
  if (!profile || !profile.id) return fail("github");
  const emails = await gh("/user/emails");
  const primary = Array.isArray(emails) ? emails.find((e) => e.primary && e.verified) : null;

  const status = env.AUTO_APPROVE === "true" ? "approved" : "waitlist";
  const user = await env.DB.prepare(
    "INSERT INTO users (github_id, login, name, email, status, created_at) VALUES (?, ?, ?, ?, ?, ?) " +
    "ON CONFLICT(github_id) DO UPDATE SET login = excluded.login, name = excluded.name, email = excluded.email RETURNING id, status"
  ).bind(profile.id, String(profile.login || ""), profile.name || null, primary ? primary.email : null, status, now()).first();
  if (!user) return fail("account");
  if (user.status === "blocked") return fail("blocked");

  const session = randomBase62(43);
  await env.DB.batch([
    env.DB.prepare("DELETE FROM sessions WHERE expires_at <= ?").bind(now()),
    env.DB.prepare("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)")
      .bind(await sha256hex(session), user.id, now() + SESSION_DAYS * 86400),
  ]);
  return redirect("/dashboard/", [setCookie(OAUTH_COOKIE, "", 0), setCookie(SESSION_COOKIE, session, SESSION_DAYS * 86400)]);
}
