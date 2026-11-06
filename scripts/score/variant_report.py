"""One variant, all four Decision Index parts, scored by the same rules as the others (training/practice data only).

usage: python scripts/score/variant_report.py PROFILE [--save]      e.g. qwen3.5-2b
Reads data/<PROFILE>/ from transfer_run.py: wg.json (+ wg_natural_b1/b2.json), w2c.json, mmlu_proxy.json,
omni_public.json, hle.json, hle_practice.json. --save writes debussy/heads_<PROFILE>.json (WildGuard heads, When2Call
head, "I don't know" threshold, and the letter-bias correction if the nested-CV rule adopts it).
"""
import os, sys, json, re, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from debussy.paths import ROOT

os.chdir(ROOT)
P = sys.argv[1]; D = f"data/{P}"; SAVE = "--save" in sys.argv; HEADS = f"debussy/heads_{P}.json"
run = lambda args: subprocess.run([sys.executable] + args, capture_output=True, text=True).stdout
print(f"=== {P}")
if SAVE and not os.path.exists(HEADS): json.dump({}, open(HEADS, "w"))

# WildGuard: calibration = all_rows() calibration (468) + natural-mix batches (170 + 80), test = 500 rows (as for K2)
wg = json.load(open(f"{D}/wg.json")); tr = [r for r in wg if r["split"] == "tr"]; te = [r for r in wg if r["split"] == "te"]
for b in (1, 2):
    f = f"{D}/wg_natural_b{b}.json"
    if os.path.exists(f): tr += json.load(open(f))
comb = f"{D}/wg_all_{len(tr)}cal_{len(te)}te.json"; json.dump(tr + te, open(comb, "w"))
out = run(["scripts/score/wildguard.py", comb, "--natural-weights"] + (["--save", "--heads", HEADS] if SAVE else []))
print(f"WildGuard ({len(tr)} cal, {len(te)} test):", re.search(r"AVERAGE F1.*?([\d.]+)\s*$", out, re.M).group(1))
for l in out.splitlines():
    if "chosen=" in l: print("   ", l.split("train-CV")[0].strip(), "->", l.split("chosen=")[1])

# When2Call: heads on the training pairs, model-only accuracy on the test items
out = run(["scripts/score/when2call.py", f"{D}/w2c.json"] + (["--save", "--heads", HEADS] if SAVE else []))
m = re.search(r"B model only\s+acc=([\d.]+)", out); print("When2Call (model only):", f"{float(m.group(1)) * 100:.1f}%" if m else out[-300:])

# Knowledge: the "I don't know" threshold is chosen on the 400 practice questions, then applied once to the stand-in
def comp(r, t):
    Pm = np.array([x["p"] for x in r]); a = np.array([x["ans"] for x in r]); ans = Pm.max(1) >= t; top = Pm.argmax(1)
    corr = (ans & (top == a)).mean() * 100; wrong = (ans & (top != a)).sum(); idk = (~ans).sum()
    nh = 100 * (1 - wrong / max(1, wrong + idk)); return corr, nh, (corr + nh) / 2, ans.mean() * 100
pr, om = json.load(open(f"{D}/mmlu_proxy.json")), json.load(open(f"{D}/omni_public.json"))
grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]; t = max(grid, key=lambda t: comp(pr, t)[2]); c, nh, sc, f = comp(om, t)
print(f"Omniscience stand-in: {sc:.1f} (threshold {t} from practice; accuracy {c:.1f}, non-halluc {nh:.1f}, answers {f:.0f}%)")
if SAVE: h = json.load(open(HEADS)); h["idk_threshold"] = t; json.dump(h, open(HEADS, "w"), indent=1)

# HLE: plain accuracy, then the letter-bias rule (nested CV on the HLE-shaped practice set only)
hl = json.load(open(f"{D}/hle.json"))
print(f"HLE-MultiChoice (raw): {np.mean([x['pred'] == x['ans'] for x in hl]) * 100:.1f}% of {len(hl)}")
if os.path.exists(f"{D}/hle_practice.json"):
    out = run(["scripts/score/hle_letter_bias.py", "--profile", P] + (["--save", "--heads", HEADS] if SAVE else []))
    for l in out.splitlines():
        if "nested CV" in l or "HLE-MultiChoice" in l: print("   ", l.strip())
