// POST /api/auth/logout: end this session.
import { SESSION_COOKIE, err, getCookie, json, sameOrigin, setCookie, sha256hex } from "../../../cloudflare/lib.js";

export async function onRequestPost({ request, env }) {
  if (!sameOrigin(request)) return err(403, "forbidden", "Cross-site request refused.");
  const token = getCookie(request, SESSION_COOKIE);
  if (token && env.DB) await env.DB.prepare("DELETE FROM sessions WHERE token_hash = ?").bind(await sha256hex(token)).run();
  return json({ ok: true }, 200, { "set-cookie": setCookie(SESSION_COOKIE, "", 0) });
}
