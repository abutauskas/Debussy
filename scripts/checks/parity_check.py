"""Check that the packaged debussy.py makes the same decisions as the research pipeline behind RESULTS.md.

usage: python scripts/checks/parity_check.py N [profile backend]   (N When2Call test items + N WildGuard test rows)
       e.g. python scripts/checks/parity_check.py 5 k2-horizon server   (needs a running llama-server with K2)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # paths below are relative to the repo root
import json, sys
import numpy as np
from debussy.model import Debussy
from debussy.skills.when2call import build_items
from debussy.skills.wildguard_rows import all_rows

n = int(sys.argv[1]); profile = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5"; backend = sys.argv[3] if len(sys.argv) > 3 else "inproc"
d = Debussy(profile=profile, backend=backend); L = ["tool_call", "request_for_info", "cannot_answer"]
REF = ({"w2c": "data/w2o2_4b_all_500tr.json", "wg": "data/wg_all_638tr.json"} if profile == "qwen3.5" else
       {"w2c": f"data/{profile}/w2c_500tr_300te.json", "wg": f"data/{profile}/wg_all_400cal.json"})
T = [json.loads(l) for l in open("data/w2c_when2call_test_mcq.jsonl")]
import random; random.Random(0).shuffle(T)
ref = [r for r in json.load(open(REF["w2c"])) if r["split"] == "te"]
h = d.h["when2call"]; agree = 0
for t, r in zip(T[:n], ref[:n]):
    p = d.when2call(t["tools"], t["question"], t["answers"])
    mine = max(L, key=lambda k: p[k])
    sig = lambda x: 1 / (1 + np.exp(-x)); lo = r["lo"]
    p_rfi = sig(h["intercept"] + sum(c * lo[f] for c, f in zip(h["coef"], h["features"])))
    theirs = "cannot_answer" if lo["exist"] < h.get("exist_cutoff", 0.0) else ("request_for_info" if p_rfi > 0.5 else "tool_call")
    agree += mine == theirs
print(f"When2Call: packaged vs pipeline decisions agree on {agree}/{n}")

R = [x for x in all_rows() if x[0] == "te"][:n]
refw = [r for r in json.load(open(REF["wg"])) if r["split"] == "te"][:n]
H = d.h["wildguard"]; agree = tot = 0; maxdiff = 0
for (_, row), r in zip(R, refw):
    p = d.wildguard(row["prompt"], row["response"])
    for k in H:
        lo = r["lo"]; hk = H[k]; x = [lo[f] for f in hk["features"]]
        sig = lambda v: 1 / (1 + np.exp(-v))
        if hk.get("interactions"):
            ph, ref_, rh = sig(lo["prompt_harm"]), sig(lo["refusal"]), sig(lo["response_harm"]); x += [ph * (1 - ref_), rh * ph, rh * (1 - ref_)]
        q = sig(hk["intercept"] + sum(c * v for c, v in zip(hk["coef"], x)) - hk["logit_threshold"])
        agree += (p[k] >= 0.5) == (q >= 0.5); tot += 1; maxdiff = max(maxdiff, abs(p[k] - q))
print(f"WildGuard: packaged vs pipeline decisions agree on {agree}/{tot} (max probability difference {maxdiff:.4f})")
