// Debussy Labs: theme switch, benchmark chart (with tooltips + table view), copy buttons.
(function () {
  // ---- theme: remembers an explicit choice; otherwise follows the system -------------------
  var root = document.documentElement, KEY = "debussy-theme";
  try { var saved = localStorage.getItem(KEY); if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved); } catch (e) {}
  function isDark() {
    var t = root.getAttribute("data-theme");
    return t ? t === "dark" : window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
    function label() { b.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme"); b.textContent = isDark() ? "☀" : "☾"; }
    label();
    b.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark"; root.setAttribute("data-theme", next);
      try { localStorage.setItem(KEY, next); } catch (e) {}
      label();
    });
  });

  // ---- copy buttons -----------------------------------------------------------------------
  document.querySelectorAll("[data-copy]").forEach(function (b) {
    b.addEventListener("click", function () {
      var src = document.getElementById(b.getAttribute("data-copy")); if (!src) return;
      var done = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1600); };
      try { navigator.clipboard.writeText(src.innerText).then(done, function () {}); } catch (e) {}
    });
  });

  // ---- benchmark chart ----------------------------------------------------------------------
  var chart = document.getElementById("bench-chart"), table = document.getElementById("bench-table"), toggle = document.getElementById("bench-toggle");
  if (!chart) return;
  var SERIES = [{ key: "duo", name: "Debussy Duo", color: "var(--s1)" }, { key: "light", name: "Debussy Light", color: "var(--s2)" }];
  var BENCH = [
    { name: "When2Call", sub: "Tool-call decisions · accuracy", duo: 93.2, light: 93.2, fmt: function (v) { return v.toFixed(1) + "%"; },
      note: "600 test questions. The Duo uses Qwen for tool calls, so both variants are the same here." },
    { name: "WildGuardTest", sub: "Safety judgements · average F1", duo: 93.6, light: 90.5, fmt: function (v) { return (v / 100).toFixed(3); },
      note: "The same 500 annotator-agreed test rows for both variants." },
    { name: "AA-Omniscience", sub: "Knowledge · ½ accuracy + ½ non-hallucination", duo: 53.1, light: 51.8, fmt: function (v) { return v.toFixed(1); },
      note: "Stand-in: Artificial Analysis's 266 public questions." },
    { name: "HLE-MultiChoice", sub: "Hard multiple choice · accuracy", duo: 15.0, light: 10.5, chance: 16.4, fmt: function (v) { return v.toFixed(1) + "%"; },
      note: "All 513 text-only questions. Chance is 16.4%." }
  ];
  var tip = document.createElement("div"); tip.className = "tooltip"; tip.setAttribute("role", "tooltip"); document.body.appendChild(tip);
  function el(tag, cls, text) { var n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
  function showTip(b, s, target) {
    tip.textContent = "";
    tip.appendChild(el("div", "tv", b.fmt(b[s.key])));
    var row = el("div", "ts"); var k = el("span", "key"); k.style.background = s.color; row.appendChild(k); row.appendChild(el("span", null, s.name + " · " + b.name)); tip.appendChild(row);
    tip.appendChild(el("div", "tn", b.note));
    var r = target.getBoundingClientRect(), w = tip.offsetWidth || 220;
    var x = Math.min(Math.max(8, r.left + Math.min(r.width, 320) - w / 2), window.innerWidth - w - 8);
    var y = r.top - tip.offsetHeight - 10; if (y < 8) y = r.bottom + 10;
    tip.style.left = x + "px"; tip.style.top = y + "px"; tip.classList.add("on");
  }
  function hideTip() { tip.classList.remove("on"); }

  var bars = el("div", "bars");
  BENCH.forEach(function (b) {
    var row = el("div", "row"); var name = el("div", "name", b.name); name.appendChild(el("span", null, b.sub)); row.appendChild(name);
    var plot = el("div", "plot"); plot.appendChild(el("div", "gridlines"));
    SERIES.forEach(function (s) {
      var hit = el("div", "bar-hit"); hit.tabIndex = 0; hit.setAttribute("aria-label", s.name + ", " + b.name + ": " + b.fmt(b[s.key]));
      var bar = el("div", "bar"); bar.style.width = b[s.key] + "%"; bar.style.background = s.color; hit.appendChild(bar);
      var val = el("span", "bar-val", b.fmt(b[s.key])); val.style.left = b[s.key] + "%"; hit.appendChild(val);
      ["pointerenter", "focus"].forEach(function (ev) { hit.addEventListener(ev, function () { showTip(b, s, hit); }); });
      ["pointerleave", "blur"].forEach(function (ev) { hit.addEventListener(ev, hideTip); });
      plot.appendChild(hit);
    });
    if (b.chance) {  // a hairline at chance level, in a layer exactly as wide as the bars' track
      var layer = el("div", "chance-layer"), c = el("div", "chance"); c.style.left = b.chance + "%";
      c.appendChild(el("span", null, "chance " + b.chance + "%")); layer.appendChild(c); plot.appendChild(layer);
      plot.classList.add("has-chance");
    }
    row.appendChild(plot); bars.appendChild(row);
  });
  chart.appendChild(bars);
  var axis = el("div", "axis"); axis.appendChild(el("div")); var ticks = el("div", "ticks");
  [0, 25, 50, 75, 100].forEach(function (t) { var s = el("span", null, String(t)); s.style.left = t + "%"; ticks.appendChild(s); });
  axis.appendChild(ticks); chart.appendChild(axis);

  chart.hidden = false; table.hidden = true;
  if (toggle) {
    toggle.hidden = false;
    toggle.addEventListener("click", function () {
      var showTable = table.hidden; table.hidden = !showTable; chart.hidden = showTable;
      toggle.textContent = showTable ? "Show chart" : "Show table"; toggle.setAttribute("aria-pressed", String(showTable));
    });
  }
  window.addEventListener("scroll", hideTip, { passive: true });
})();
