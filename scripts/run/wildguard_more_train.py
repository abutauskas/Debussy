"""Add more WildGuard TRAIN (calibration) rows, excluding ones already used. Bounded, resumable.

usage: python wg_more_train.py MODEL.gguf EXISTING.json N OUT.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
import json, os, sys, time
import pandas as pd
from debussy.skills.wildguard import Judge, D, COL, POS

path, existing, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tr = pd.read_parquet(D + "wg_wildguard_train.parquet").dropna(subset=["response_harm_label", "response_refusal_label"])
keys = ["prompt_harm_label", "response_refusal_label", "response_harm_label"]
# rebuild the original 120-row calibration sample so we can exclude it
idx0 = [i for _, g in tr.groupby(keys) for i in g.sample(min(len(g), 120 // 6 + 1), random_state=0).index]
used = set(tr.loc[idx0].sample(frac=1, random_state=0).head(120).index)
seed = int(os.environ.get("WG_SEED", "1"))  # batch 1 used seed 1; later batches also exclude every earlier batch
rest = tr.drop(index=list(used))
for s in range(1, seed + 1):
    idx = [i for _, g in rest.groupby(keys) for i in g.sample(min(len(g), n // 6 + 1), random_state=s).index]
    new = rest.loc[idx].sample(frac=1, random_state=s).head(n)
    if s < seed:
        rest = rest.drop(index=list(new.index))
rows = json.load(open(out)) if os.path.exists(out) else []
j = Judge(path); t0 = time.time(); s0 = len(rows); items = list(new.iterrows())
for i in range(s0, len(items)):
    _, r = items[i]
    rows.append({"split": "tr", "lo": j.probs(r["prompt"], r["response"]),
                 "y": {k: (None if pd.isna(r[c]) else int(r[c] == POS[k])) for k, c in COL.items()}})
    if (i + 1) % 20 == 0 or i + 1 == len(items):
        json.dump(rows, open(out, "w")); print(f"{i+1}/{len(items)} {(time.time()-t0)/(i+1-s0):.2f}s/ex", flush=True)
print("DONE")
