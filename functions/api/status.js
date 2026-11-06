// GET /api/status: whether sign-ups are open (the sign-in page shows "coming soon" otherwise).
import { json, signupsOpen } from "../../cloudflare/lib.js";

export async function onRequestGet({ env }) {
  return json({ signups_open: signupsOpen(env) });
}
