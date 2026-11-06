# Stand-in for GitHub's OAuth token endpoint and user API (local testing only).
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj, code=200):
        d = json.dumps(obj).encode(); self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        ok = body.get("client_secret") == "test-secret" and body.get("code") == "good-code"
        self._send({"access_token": "gho_test", "token_type": "bearer"} if ok else {"error": "bad_verification_code"})
    def do_GET(self):
        if self.headers.get("authorization") != "Bearer gho_test": return self._send({"message": "Bad credentials"}, 401)
        if self.path == "/user": return self._send({"id": 4242, "login": "testuser", "name": "Test User"})
        if self.path == "/user/emails": return self._send([{"email": "other@example.com", "primary": False, "verified": True},
                                                           {"email": "test@example.com", "primary": True, "verified": True}])
        self._send({}, 404)
HTTPServer(("127.0.0.1", 8012), H).serve_forever()
