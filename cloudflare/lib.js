// Shared helpers for the debussylabs.com Pages Functions (functions/). Lives outside functions/ so it is not a route.

export const SESSION_COOKIE = "__Host-dbsy_session";
export const OAUTH_COOKIE = "__Host-dbsy_oauth";
export const SESSION_DAYS = 30;
export const KEY_RE = /^dbsy_[A-Za-z0-9]{40}$/;

export function json(obj, status = 200, headers = {}) {
  return new Response(JSON.stringify(obj), {
    status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...headers },
  });
}
export const err = (status, type, message, headers) => json({ error: { type, message } }, status, headers);
export const now = () => Math.floor(Date.now() / 1000);

const B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
/** n random base62 characters from the platform CSPRNG (rejection sampling, no modulo bias). */
export function randomBase62(n) {
  let out = "";
  while (out.length < n) {
    const buf = crypto.getRandomValues(new Uint8Array(n * 2));
    for (const b of buf) { if (b < 248 && out.length < n) out += B62[b % 62]; }
  }
  return out;
}
export async function sha256hex(s) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function getCookie(request, name) {
  const h = request.headers.get("cookie") || "";
  for (const part of h.split(";")) {
    const i = part.indexOf("=");
    if (i > 0 && part.slice(0, i).trim() === name) return part.slice(i + 1).trim();
  }
  return null;
}
export const setCookie = (name, value, maxAge) => `${name}=${value}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${maxAge}`;

/** State-changing requests must come from our own pages (with SameSite=Lax cookies, this blocks cross-site forgery). */
export function sameOrigin(request) {
  const origin = request.headers.get("origin");
  return origin !== null && origin === new URL(request.url).origin;
}

export function dbReady(env) { return Boolean(env && env.DB); }

/** The signed-in user for this request, or null. */
export async function currentUser(env, request) {
  const token = getCookie(request, SESSION_COOKIE);
  if (!token || !dbReady(env)) return null;
  return env.DB.prepare(
    "SELECT u.id, u.login, u.name, u.email, u.status, u.created_at FROM sessions s JOIN users u ON u.id = s.user_id " +
    "WHERE s.token_hash = ? AND s.expires_at > ?"
  ).bind(await sha256hex(token), now()).first();
}

export function redirect(location, cookies = []) {
  const h = new Headers({ location, "cache-control": "no-store" });
  for (const c of cookies) h.append("set-cookie", c);
  return new Response(null, { status: 302, headers: h });
}

export const intEnv = (env, name, dflt) => { const v = parseInt((env && env[name]) || "", 10); return Number.isFinite(v) && v > 0 ? v : dflt; };
