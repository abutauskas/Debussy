// GET /api/keys: list your keys. POST /api/keys {"name"}: create one; the full key is returned ONCE and stored only hashed.
import { currentUser, err, intEnv, json, now, randomBase62, sameOrigin, sha256hex } from "../../../cloudflare/lib.js";

export async function onRequestGet({ request, env }) {
  if (!env.DB) return err(503, "unavailable", "Accounts are not set up yet.");
  const u = await currentUser(env, request);
  if (!u) return err(401, "unauthorized", "Not signed in.");
  const { results } = await env.DB.prepare(
    "SELECT id, name, prefix, created_at, last_used_at, revoked_at FROM api_keys WHERE user_id = ? ORDER BY created_at DESC"
  ).bind(u.id).all();
  return json({ keys: results });
}

export async function onRequestPost({ request, env }) {
  if (!env.DB) return err(503, "unavailable", "Accounts are not set up yet.");
  if (!sameOrigin(request)) return err(403, "forbidden", "Cross-site request refused.");
  const u = await currentUser(env, request);
  if (!u) return err(401, "unauthorized", "Not signed in.");
  if (u.status !== "approved") return err(403, "not_approved", "Your account is on the waitlist. We'll email you when API access opens.");
  let name = "";
  try { name = String(((await request.json()) || {}).name || "").trim(); } catch (e) { /* empty body is fine */ }
  name = (name || "My key").slice(0, 60);
  const active = await env.DB.prepare("SELECT COUNT(*) AS n FROM api_keys WHERE user_id = ? AND revoked_at IS NULL").bind(u.id).first();
  const max = intEnv(env, "MAX_KEYS", 5);
  if (active && active.n >= max) return err(409, "too_many_keys", `You can have up to ${max} active keys. Revoke one first.`);
  const key = "dbsy_" + randomBase62(40);
  const row = await env.DB.prepare(
    "INSERT INTO api_keys (user_id, name, prefix, key_hash, created_at) VALUES (?, ?, ?, ?, ?) RETURNING id, name, prefix, created_at"
  ).bind(u.id, name, key.slice(0, 9), await sha256hex(key), now()).first();
  return json({ ...row, key }, 201);
}
