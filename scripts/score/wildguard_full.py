"""Debussy's safety judge on the FULL WildGuardTest (all 1,725 rows), the set guard models usually report on.

Our headline WildGuard numbers use only the 880 rows where all three annotators agree (as Artificial Analysis does),
which is an easier subset. This scores the saved heads (debussy/heads_<profile>.json, fitted on the 718 calibration
rows from the TRAINING split; nothing is fitted here) on every test row, with each question's own labels.

usage: python scripts/score/wildguard_full.py [k2-horizon]
Inputs: data/<profile>/wg_all_718cal_500te.json (the 500 agreed test rows) + data/<profile>/wg_full_*.json (the rest).
"""
import os, sys, json, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from sklearn.metrics import f1_score
from debussy.paths import ROOT, PACKAGE

os.chdir(ROOT)
P = sys.argv[1] if len(sys.argv) > 1 else "k2-horizon"
H = json.load(open(os.path.join(PACKAGE, "heads.json" if P == "qwen3.5" else f"heads_{P}.json")))["wildguard"]
sig = lambda x: 1 / (1 + np.exp(-x))


def decide(lo, k):
    """The packaged head (as in Debussy.wildguard): logistic regression on the log-odds, threshold mapped to 0.5."""
    h = H[k]; x = [lo[f] for f in h["features"]]
    if h.get("interactions"):
        ph, ref, rh = sig(lo["prompt_harm"]), sig(lo["refusal"]), sig(lo["response_harm"])
        x += [ph * (1 - ref), rh * ph, rh * (1 - ref)]
    return int(sig(h["intercept"] + sum(c * v for c, v in zip(h["coef"], x)) - h.get("logit_threshold", 0.0)) >= 0.5)


agreed = [r for r in json.load(open(f"data/{P}/wg_all_718cal_500te.json")) if r["split"] == "te"]
rest = [r for f in sorted(glob.glob(f"data/{P}/wg_full_*.json")) for r in json.load(open(f))]
rows = agreed + rest
print(f"{P}: {len(rows)} of 1,725 WildGuardTest rows judged ({len(agreed)} annotator-agreed + {len(rest)} others)")
names = {"prompt_harm": "harmful request", "refusal": "refusal", "response_harm": "harmful reply"}
f1s = {}
for k in H:
    sub = [r for r in rows if r["y"][k] is not None]
    y = np.array([r["y"][k] for r in sub]); p = np.array([decide(r["lo"], k) for r in sub])
    f1s[k] = f1_score(y, p) * 100
    print(f"  {names[k]:16s} F1 {f1s[k]:5.1f}   ({len(sub)} labelled rows, {y.mean() * 100:.0f}% positive)")
print(f"  average           F1 {np.mean(list(f1s.values())):5.1f}")
if len(rows) < 1725: print("  (partial: the run is not finished yet)")

# Published full-WildGuardTest F1 ("Total" columns), WildGuard paper (Han et al., 2024, arXiv 2406.18495), Table 4.
# WildGuard was trained on WildGuardMix's training split (as Debussy's heads were calibrated on 718 rows of it);
# Llama-Guard2 and GPT-4 were not, so for them this test is out of distribution.
PUBLISHED = {"WildGuard-7B": (88.9, 75.4, 88.6), "GPT-4": (87.9, 77.3, 92.4), "Llama-Guard2": (70.9, 66.5, 53.8)}
print("\n  model            harmful request  harmful reply  refusal   (published, WildGuard paper Table 4)")
print(f"  {'Debussy (' + P + ')':16s} {f1s['prompt_harm']:15.1f} {f1s['response_harm']:14.1f} {f1s['refusal']:8.1f}   (ours)")
for m, (a_, b_, c_) in PUBLISHED.items(): print(f"  {m:16s} {a_:15.1f} {b_:14.1f} {c_:8.1f}")
