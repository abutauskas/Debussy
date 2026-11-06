// GET /api/me: the signed-in account (used by the dashboard).
import { SESSION_COOKIE, currentUser, err, json, sameOrigin, setCookie } from "../../cloudflare/lib.js";

export async function onRequestGet({ request, env }) {
  if (!env.DB) return err(503, "unavailable", "Accounts are not set up yet.");
  const u = await currentUser(env, request);
  if (!u) return err(401, "unauthorized", "Not signed in.");
  return json({ login: u.login, name: u.name, email: u.email, status: u.status, created_at: u.created_at });
}

// DELETE /api/me: delete your account, keys, sessions and usage counts. Cannot be undone.
export async function onRequestDelete({ request, env }) {
  if (!env.DB) return err(503, "unavailable", "Accounts are not set up yet.");
  if (!sameOrigin(request)) return err(403, "forbidden", "Cross-site request refused.");
  const u = await currentUser(env, request);
  if (!u) return err(401, "unauthorized", "Not signed in.");
  await env.DB.batch([
    env.DB.prepare("DELETE FROM usage WHERE key_id IN (SELECT id FROM api_keys WHERE user_id = ?)").bind(u.id),
    env.DB.prepare("DELETE FROM api_keys WHERE user_id = ?").bind(u.id),
    env.DB.prepare("DELETE FROM sessions WHERE user_id = ?").bind(u.id),
    env.DB.prepare("DELETE FROM users WHERE id = ?").bind(u.id),
  ]);
  return json({ ok: true }, 200, { "set-cookie": setCookie(SESSION_COOKIE, "", 0) });
}
