// Debussy Labs account pages (sign in, dashboard). Talks to /api/* (Cloudflare Pages Functions in functions/).
// Everything that comes from an account (names, key names) is inserted with textContent, never as HTML.
(function () {
  var $ = function (s) { return document.querySelector(s); };
  function api(path, opts) {
    opts = opts || {};
    return fetch(path, { method: opts.method || "GET", credentials: "same-origin",
                         headers: opts.body ? { "content-type": "application/json" } : {}, body: opts.body })
      .then(function (r) { return r.json().catch(function () { return null; }).then(function (b) { return { status: r.status, body: b }; }); })
      .catch(function () { return { status: 0, body: null }; });
  }
  function el(tag, cls, text) { var n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
  function day(ts) { return ts ? new Date(ts * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "never"; }
  function errMsg(res, fallback) { return (res.body && res.body.error && res.body.error.message) || fallback; }

  // ---- sign-in page --------------------------------------------------------------------------
  if ($("#login")) {
    var ERR = { state: "The sign-in expired or was interrupted. Please try again.", github: "GitHub didn't confirm the sign-in. Please try again.",
                blocked: "This account can't sign in.", unavailable: "Sign-in isn't open yet.", closed: "Sign-ups aren't open yet.",
                account: "We couldn't create your account. Please try again." };
    var code = new URLSearchParams(location.search).get("error");
    if (code) { var n = $("#login-error"); n.hidden = false; n.textContent = ERR[code] || "Sign-in didn't work. Please try again."; }
    api("/api/me").then(function (r) { if (r.status === 200) location.replace("/dashboard/"); });
    api("/api/status").then(function (r) {   // the page says "coming soon" unless sign-ups are switched on
      if (!(r.status === 200 && r.body && r.body.signups_open)) return;
      $("#login-title").textContent = "Sign in";
      $("#login-lede").textContent = "Use your GitHub account. Signing in for the first time creates your Debussy Labs account.";
      $("#selfhost-btn").hidden = true; $("#github-btn").hidden = false; $("#login-fine").hidden = false;
    });
    return;
  }

  // ---- dashboard -----------------------------------------------------------------------------
  var dash = $("#dashboard"); if (!dash) return;
  var state = { keys: [] };

  function renderAccount(me) {
    var who = $("#who"); who.textContent = "";
    var av = el("div", "avatar", (me.login || "?").slice(0, 1).toUpperCase()); who.appendChild(av);
    var t = el("div"); t.appendChild(el("div", null, me.name || me.login)); var sub = el("div", "small muted", "@" + me.login + (me.email ? " · " + me.email : "")); t.appendChild(sub); who.appendChild(t);
    var pill = $("#status-pill");
    pill.className = "pill " + (me.status === "approved" ? "ok" : me.status === "waitlist" ? "wait" : "off");
    pill.textContent = me.status === "approved" ? "API access" : me.status === "waitlist" ? "On the waitlist" : "No access";
    $("#waitlist").hidden = me.status !== "waitlist";
    $("#blocked").hidden = me.status !== "blocked";
    $("#keys-panel").hidden = me.status !== "approved";
  }

  function renderKeys() {
    var body = $("#keys-body"); body.textContent = "";
    var active = state.keys.filter(function (k) { return !k.revoked_at; });
    $("#keys-empty").hidden = state.keys.length > 0;
    $("#keys-table").hidden = state.keys.length === 0;
    $("#key-count").textContent = active.length + " active";
    state.keys.forEach(function (k) {
      var tr = el("tr");
      tr.appendChild(el("td", null, k.name));
      tr.appendChild(el("td", "num", k.prefix + "…"));
      tr.appendChild(el("td", null, day(k.created_at)));
      tr.appendChild(el("td", null, day(k.last_used_at)));
      var td = el("td");
      if (k.revoked_at) td.appendChild(el("span", "pill off", "Revoked"));
      else {
        var b = el("button", "btn btn-ghost btn-small btn-danger", "Revoke"); b.type = "button";
        b.addEventListener("click", function () {
          if (!confirm("Revoke “" + k.name + "”? Apps using it stop working immediately.")) return;
          api("/api/keys/" + k.id, { method: "DELETE" }).then(function (r) { if (r.status === 200) loadKeys(); else alert(errMsg(r, "Couldn't revoke the key.")); });
        });
        td.appendChild(b);
      }
      tr.appendChild(td); body.appendChild(tr);
    });
  }

  function loadKeys() { return api("/api/keys").then(function (r) { if (r.status === 200) { state.keys = r.body.keys || []; renderKeys(); } }); }

  $("#create-key").addEventListener("submit", function (e) {
    e.preventDefault();
    var name = $("#key-name").value.trim(), btn = $("#create-btn"), msg = $("#create-msg");
    btn.disabled = true; msg.hidden = true;
    api("/api/keys", { method: "POST", body: JSON.stringify({ name: name }) }).then(function (r) {
      btn.disabled = false;
      if (r.status !== 201) { msg.hidden = false; msg.textContent = errMsg(r, "Couldn't create the key."); return; }
      $("#key-name").value = "";
      $("#newkey").hidden = false; $("#newkey-value").textContent = r.body.key; $("#newkey-name").textContent = r.body.name;
      loadKeys();
    });
  });
  $("#copy-key").addEventListener("click", function () {
    var b = this; try { navigator.clipboard.writeText($("#newkey-value").textContent).then(function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1500); }); } catch (e) {}
  });
  $("#hide-key").addEventListener("click", function () { $("#newkey").hidden = true; $("#newkey-value").textContent = ""; });
  $("#signout").addEventListener("click", function () { api("/api/auth/logout", { method: "POST" }).then(function () { location.href = "/"; }); });
  $("#delete-account").addEventListener("click", function () {
    if (!confirm("Delete your Debussy Labs account? Your keys stop working and your account data is deleted. This can't be undone.")) return;
    api("/api/me", { method: "DELETE" }).then(function (r) { if (r.status === 200) location.href = "/"; else alert(errMsg(r, "Couldn't delete the account.")); });
  });

  api("/api/me").then(function (r) {
    if (r.status === 401) { location.replace("/login/"); return; }
    if (r.status !== 200) { $("#loading").textContent = errMsg(r, "Accounts aren't open yet. Please check back soon."); return; }
    $("#loading").hidden = true; $("#account").hidden = false;
    var nb = document.querySelector('.nav a[href="/login/"]'); if (nb) { nb.textContent = "Dashboard"; nb.setAttribute("href", "/dashboard/"); }
    renderAccount(r.body);
    if (r.body.status === "approved") loadKeys();
  });
})();
