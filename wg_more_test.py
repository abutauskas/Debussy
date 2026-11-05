"""Score more WildGuard TEST rows (annotator-agreed subset), continuing the same fixed sample. Bounded, resumable.

usage: python wg_more_test.py MODEL.gguf START END OUT.json   (rows START..END of te.sample(END, random_state=0))
"""
import json, os, sys, time
import pandas as pd
from wildguard_yesno import Judge, D, COL, POS

path, start, end, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
te = pd.read_parquet(D + "wg_wildguard_test.parquet")
te = te[(te.prompt_harm_agreement == 3) & (te.response_refusal_agreement == 3) & (te.response_harm_agreement == 3)]
items = list(te.sample(end, random_state=0).iloc[start:end].iterrows())
rows = json.load(open(out)) if os.path.exists(out) else []
j = Judge(path); t0 = time.time(); s0 = len(rows)
for i in range(s0, len(items)):
    _, r = items[i]
    rows.append({"split": "te", "lo": j.probs(r["prompt"], r["response"]),
                 "y": {k: (None if pd.isna(r[c]) else int(r[c] == POS[k])) for k, c in COL.items()}})
    if (i + 1) % 20 == 0 or i + 1 == len(items):
        json.dump(rows, open(out, "w")); print(f"{i+1}/{len(items)} {(time.time()-t0)/(i+1-s0):.2f}s/ex", flush=True)
print("DONE")
