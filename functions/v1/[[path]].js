// POST /v1/{wildguard,when2call,choice}: the public API. Checks the API key, applies rate limits, counts usage
// (never the request contents), and forwards the request to the Debussy API server (debussy/server.py).
import { KEY_RE, err, intEnv, now, sha256hex } from "../../cloudflare/lib.js";

const PATHS = new Set(["wildguard", "when2call", "choice"]);
const MAX_BODY = 64 * 1024;

export async function onRequest({ request, env, params }) {
  const path = (params.path || []).join("/");
  if (!PATHS.has(path)) return err(404, "not_found", "Unknown endpoint. Use /v1/wildguard, /v1/when2call or /v1/choice.");
  if (request.method !== "POST") return err(405, "method_not_allowed", "Use POST.", { allow: "POST" });
  if (!env.DB) return err(503, "unavailable", "The hosted API is not live yet.");

  const m = (request.headers.get("authorization") || "").match(/^Bearer\s+(\S+)$/);
  if (!m || !KEY_RE.test(m[1])) return err(401, "unauthorized", "Missing or malformed API key. Send 'Authorization: Bearer dbsy_...'.");
  const key = await env.DB.prepare(
    "SELECT k.id, u.status FROM api_keys k JOIN users u ON u.id = k.user_id WHERE k.key_hash = ? AND k.revoked_at IS NULL"
  ).bind(await sha256hex(m[1])).first();
  if (!key) return err(401, "unauthorized", "Invalid or revoked API key.");
  if (key.status !== "approved") return err(403, "not_approved", "This account does not have API access.");

  const t = now(), minute = "m:" + Math.floor(t / 60), day = "d:" + new Date(t * 1000).toISOString().slice(0, 10);
  const perMin = intEnv(env, "RATE_PER_MINUTE", 20), perDay = intEnv(env, "RATE_PER_DAY", 500);
  const bump = (b) => env.DB.prepare(
    "INSERT INTO usage (key_id, bucket, count) VALUES (?, ?, 1) ON CONFLICT(key_id, bucket) DO UPDATE SET count = count + 1 RETURNING count"
  ).bind(key.id, b);
  const stmts = [bump(minute), bump(day), env.DB.prepare("UPDATE api_keys SET last_used_at = ? WHERE id = ?").bind(t, key.id)];
  if (Math.random() < 0.02) {  // housekeeping: minute counters after 2 hours, daily counters after 90 days
    stmts.push(env.DB.prepare("DELETE FROM usage WHERE bucket LIKE 'm:%' AND bucket < ?").bind("m:" + (Math.floor(t / 60) - 120)));
    stmts.push(env.DB.prepare("DELETE FROM usage WHERE bucket LIKE 'd:%' AND bucket < ?").bind("d:" + new Date((t - 90 * 86400) * 1000).toISOString().slice(0, 10)));
  }
  const res = await env.DB.batch(stmts);
  const usedMin = res[0].results[0].count, usedDay = res[1].results[0].count;
  const limits = { "x-ratelimit-limit-day": String(perDay), "x-ratelimit-remaining-day": String(Math.max(0, perDay - usedDay)) };
  if (usedMin > perMin) return err(429, "rate_limited", `Limit: ${perMin} requests per minute.`, { ...limits, "retry-after": String(60 - (t % 60)) });
  if (usedDay > perDay) return err(429, "rate_limited", `Limit: ${perDay} requests per day.`, limits);

  if (!env.INFERENCE_URL) return err(503, "unavailable", "The hosted API is not live yet. You can run Debussy yourself: python -m debussy.server", limits);
  if (parseInt(request.headers.get("content-length") || "0", 10) > MAX_BODY) return err(413, "too_large", "Request bodies are limited to 64 KB.", limits);
  const body = await request.arrayBuffer();
  if (body.byteLength > MAX_BODY) return err(413, "too_large", "Request bodies are limited to 64 KB.", limits);

  let upstream;
  try {
    upstream = await fetch(env.INFERENCE_URL.replace(/\/$/, "") + "/v1/" + path, {
      method: "POST", body,
      headers: { "content-type": "application/json", "x-debussy-gateway-token": env.INFERENCE_TOKEN || "" },
    });
  } catch (e) {
    return err(502, "upstream_error", "The model server did not answer. Please try again.", limits);
  }
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...limits },
  });
}
