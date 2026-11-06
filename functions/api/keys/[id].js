// DELETE /api/keys/:id: revoke one of your keys (it stops working immediately).
import { currentUser, err, json, now, sameOrigin } from "../../../cloudflare/lib.js";

export async function onRequestDelete({ request, env, params }) {
  if (!env.DB) return err(503, "unavailable", "Accounts are not set up yet.");
  if (!sameOrigin(request)) return err(403, "forbidden", "Cross-site request refused.");
  const u = await currentUser(env, request);
  if (!u) return err(401, "unauthorized", "Not signed in.");
  const id = parseInt(params.id, 10);
  if (!Number.isFinite(id)) return err(400, "invalid_request", "Bad key id.");
  const r = await env.DB.prepare("UPDATE api_keys SET revoked_at = ? WHERE id = ? AND user_id = ? AND revoked_at IS NULL")
    .bind(now(), id, u.id).run();
  if (!r.meta || r.meta.changes !== 1) return err(404, "not_found", "No such active key.");
  return json({ ok: true });
}
