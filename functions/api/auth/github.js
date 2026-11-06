// GET /api/auth/github: start "Sign in with GitHub" (OAuth web flow with a one-time state cookie).
import { OAUTH_COOKIE, randomBase62, redirect, setCookie, signupsOpen, siteOrigin } from "../../../cloudflare/lib.js";

export async function onRequestGet({ request, env }) {
  if (!signupsOpen(env)) return redirect("/login/?error=closed");
  // sign in on the main address only (www and *.pages.dev are sent there first), so the callback matches GitHub's
  const origin = siteOrigin(env, request);
  if (new URL(request.url).origin !== origin) return redirect(origin + "/api/auth/github");
  const state = randomBase62(32);
  const url = new URL("https://github.com/login/oauth/authorize");
  url.searchParams.set("client_id", env.GITHUB_CLIENT_ID);
  url.searchParams.set("redirect_uri", origin + "/api/auth/callback");
  url.searchParams.set("scope", "read:user user:email");
  url.searchParams.set("state", state);
  return redirect(url.toString(), [setCookie(OAUTH_COOKIE, state, 600)]);
}
