"""Add WildGuard TRAIN calibration rows drawn at random (natural label mix), excluding all rows used before.

usage: WG_CLIP=... python wg_natural_train.py N OUT.json     (resumable)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
import json, os, sys, time
import pandas as pd
from debussy.skills.wildguard_rows import all_rows, D
from debussy.skills.wildguard import Judge, COL, POS
from debussy.paths import ROOT as _ROOT

from debussy.skills.wildguard_rows import natural_rows

n, out = int(sys.argv[1]), sys.argv[2]
new = natural_rows(n, int(os.environ.get("NAT_BATCH", "1")))  # batch 1 = 170 rows (seed 7); later batches exclude earlier ones
rows = json.load(open(out)) if os.path.exists(out) else []
j = Judge(_ROOT + "/models/Qwen3.5-4B-Q4_K_M.gguf"); t0 = time.time(); s0 = len(rows)
items = list(new.iterrows())
for i in range(s0, len(items)):
    _, r = items[i]
    rows.append({"split": "tr", "natural": True, "lo": j.probs(r["prompt"], r["response"]),
                 "y": {k: (None if pd.isna(r[c]) else int(r[c] == POS[k])) for k, c in COL.items()}})
    if (i + 1) % 20 == 0 or i + 1 == len(items):
        json.dump(rows, open(out, "w")); print(f"{i+1}/{len(items)} {(time.time()-t0)/(i+1-s0):.2f}s/ex", flush=True)
print("DONE")
