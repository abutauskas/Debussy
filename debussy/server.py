"""Debussy HTTP API: the three decisions as small JSON endpoints, for self-hosting and behind the debussylabs.com gateway.

usage: python -m debussy.server [--variant duo|light|k2] [--host 127.0.0.1] [--port 8000] [--fake]

  POST /v1/wildguard  {"prompt": str, "response": str}
       -> {"model": ..., "probabilities": {"prompt_harm": p, "refusal": p, "response_harm": p}}
  POST /v1/when2call  {"tools": [tool, ...], "request": str,
                       "candidates": {"tool_call": str|object, "request_for_info": str, "cannot_answer": str, "direct": str}}
       -> {"model": ..., "probabilities": {"tool_call": p, "request_for_info": p, "cannot_answer": p, "direct": p}}
  POST /v1/choice     {"question": str, "options": [str, ...], "idk_option": str (optional)}
       -> {"model": ..., "probabilities": {option: p, ...}}
  GET  /health        -> {"status": "ok", "model": ...}

- One model, one decision at a time (a lock); every decision is a single pass.
- If DEBUSSY_GATEWAY_TOKEN is set, /v1 requests must carry it in the X-Debussy-Gateway-Token header, so a hosted server
  only answers the debussylabs.com gateway (which checks users' API keys and rate limits).
- Request bodies over 64 KB are refused. Request contents are never logged.
- --fake answers with fixed probabilities and loads no model (for testing the website and gateway).
"""
import argparse, hmac, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 64 * 1024
VERSION = "0.1"


class BadRequest(Exception):
    pass


def _str(body, key, limit=20000, required=True):
    v = body.get(key)
    if v is None and not required: return None
    if not isinstance(v, str) or not v.strip(): raise BadRequest(f"'{key}' must be a non-empty string")
    if len(v) > limit: raise BadRequest(f"'{key}' is longer than {limit} characters")
    return v


def _validate(path, body):
    """Returns the keyword arguments for the decision, or raises BadRequest. Pure function (tested without a model)."""
    if not isinstance(body, dict): raise BadRequest("the request body must be a JSON object")
    if path == "/v1/wildguard":
        return {"prompt": _str(body, "prompt"), "response": _str(body, "response")}
    if path == "/v1/when2call":
        tools = body.get("tools")
        if not isinstance(tools, list) or not 1 <= len(tools) <= 64 or not all(isinstance(t, (dict, str)) for t in tools):
            raise BadRequest("'tools' must be a list of 1-64 tool definitions (objects or JSON strings)")
        c = body.get("candidates")
        if not isinstance(c, dict) or "tool_call" not in c:
            raise BadRequest("'candidates' must be an object with at least 'tool_call' (and ideally 'request_for_info')")
        cands = {}
        for k in ("tool_call", "request_for_info", "cannot_answer", "direct"):
            v = c.get(k, "")
            if isinstance(v, (dict, list)): v = json.dumps(v, ensure_ascii=False)
            if not isinstance(v, str) or len(v) > 8000: raise BadRequest(f"'candidates.{k}' must be a string or JSON object under 8000 characters")
            cands[k] = v
        return {"tools": tools, "user_msg": _str(body, "request"), "candidates": cands}
    if path == "/v1/choice":
        opts = body.get("options")
        if not isinstance(opts, list) or not 2 <= len(opts) <= 26 or not all(isinstance(o, str) and o.strip() and len(o) <= 2000 for o in opts):
            raise BadRequest("'options' must be a list of 2-26 non-empty strings")
        if len(set(opts)) != len(opts): raise BadRequest("'options' must not repeat")
        idk = _str(body, "idk_option", 200, required=False)
        if idk is not None and idk not in opts: raise BadRequest("'idk_option' must be one of 'options'")
        return {"question": _str(body, "question"), "options": opts, "idk_option": idk}
    raise KeyError(path)


class FakeModel:
    """Fixed answers, no model: for testing the website, gateway and clients."""
    def wildguard(self, prompt, response): return {"prompt_harm": 0.5, "refusal": 0.5, "response_harm": 0.5}
    def when2call(self, tools, user_msg, candidates): return {"tool_call": 0.25, "request_for_info": 0.25, "cannot_answer": 0.25, "direct": 0.25}
    def choice(self, question, options, idk_option=None): return {o: 1 / len(options) for o in options}


def load_model(variant):
    from debussy.model import Debussy, DebussyDuo
    if variant == "duo": return DebussyDuo()
    if variant == "k2": return Debussy(profile="k2-horizon", backend="server")
    return Debussy()


def make_handler(model, model_name, gateway_token=None):
    lock = threading.Lock()
    calls = {"/v1/wildguard": model.wildguard, "/v1/when2call": model.when2call, "/v1/choice": model.choice}

    class Handler(BaseHTTPRequestHandler):
        server_version = "Debussy/" + VERSION; sys_version = ""

        def log_message(self, fmt, *args):  # method, path and status only; never request bodies
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}\n")

        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store"); self.end_headers()
            self.wfile.write(data)

        def _error(self, code, kind, msg): self._send(code, {"error": {"type": kind, "message": msg}})

        def do_GET(self):
            if self.path.split("?")[0] == "/health": return self._send(200, {"status": "ok", "model": model_name})
            self._error(404, "not_found", "unknown path")

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in calls: return self._error(404, "not_found", "unknown path; use /v1/wildguard, /v1/when2call or /v1/choice")
            if gateway_token and not hmac.compare_digest(self.headers.get("X-Debussy-Gateway-Token", ""), gateway_token):
                return self._error(401, "unauthorized", "missing or wrong gateway token")
            try: n = int(self.headers.get("Content-Length", "0"))
            except ValueError: n = -1
            if n < 0 or n > MAX_BODY: return self._error(413, "too_large", f"request bodies are limited to {MAX_BODY} bytes")
            try:
                kw = _validate(path, json.loads(self.rfile.read(n) or b"null"))
            except json.JSONDecodeError: return self._error(400, "invalid_request", "the body is not valid JSON")
            except BadRequest as e: return self._error(400, "invalid_request", str(e))
            t0 = time.time()
            try:
                with lock: probs = calls[path](**kw)
            except Exception as e:  # keep details out of the response
                sys.stderr.write(f"decision failed: {type(e).__name__}\n"); return self._error(500, "server_error", "the decision failed")
            self._send(200, {"model": model_name, "probabilities": {k: round(float(v), 6) for k, v in probs.items()},
                             "latency_ms": int((time.time() - t0) * 1000)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Debussy HTTP API")
    ap.add_argument("--variant", choices=["duo", "light", "k2"], default=os.environ.get("DEBUSSY_VARIANT", "light"))
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fake", action="store_true", help="fixed answers, no model (testing only)")
    a = ap.parse_args()
    model = FakeModel() if a.fake else load_model(a.variant)
    name = f"debussy-{a.variant}-{VERSION}" + ("-fake" if a.fake else "")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(model, name, os.environ.get("DEBUSSY_GATEWAY_TOKEN") or None))
    print(f"Debussy API ({name}) on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
