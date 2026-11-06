// GET /api/auth/github: start "Sign in with GitHub" (OAuth web flow with a one-time state cookie).
import { OAUTH_COOKIE, err, randomBase62, redirect, setCookie } from "../../../cloudflare/lib.js";

export async function onRequestGet({ request, env }) {
  if (!env.GITHUB_CLIENT_ID || !env.GITHUB_CLIENT_SECRET || !env.DB) return err(503, "unavailable", "Sign-in is not set up yet.");
  const state = randomBase62(32);
  const url = new URL("https://github.com/login/oauth/authorize");
  url.searchParams.set("client_id", env.GITHUB_CLIENT_ID);
  url.searchParams.set("redirect_uri", new URL("/api/auth/callback", request.url).toString());
  url.searchParams.set("scope", "read:user user:email");
  url.searchParams.set("state", state);
  return redirect(url.toString(), [setCookie(OAUTH_COOKIE, state, 600)]);
}
