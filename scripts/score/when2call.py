"""Score option-aware When2Call results. Usage: python scripts/score/when2call.py RESULTS.json [--save]

Variant A ("tool list check"): cannot_answer iff the proposed call's function is not in the tool list (string match).
Variant B ("model only"): cannot_answer iff the model's own 'exist' log-odds < cut-off. The cut-off is chosen on TRAIN:
every training tool exists, so it is set where 99% of them pass (1st percentile, capped at 0).
Both: tool_call vs request_for_info by a logistic regression on the yes/no log-odds. The feature set is
picked by 5-fold cross-validation on TRAIN only; test rows are never used for any choice.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
import itertools, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

rows = json.load(open(sys.argv[1]))
tr = [r for r in rows if r["split"] == "tr"]; te = [r for r in rows if r["split"] == "te"]
keys = [k for k in ["args", "rfi", "req_args", "rfi_needed"] if k in rows[0]["lo"]]
F = lambda rs, ks: np.array([[r["lo"][k] for k in ks] for r in rs])
ytr = np.array([r["y"] for r in tr])
cands = [list(c) for n in range(1, len(keys) + 1) for c in itertools.combinations(keys, n)]
cv = {tuple(c): cross_val_score(LogisticRegression(max_iter=2000), F(tr, c), ytr, cv=5).mean() for c in cands}
best = list(max(cv, key=cv.get))
print(f"train {len(tr)}  test {len(te)}")
for c, s in sorted(cv.items(), key=lambda x: -x[1])[:5]: print(f"  train-CV {s:.3f}  {'+'.join(c)}")
print("chosen features:", "+".join(best))
m = LogisticRegression(max_iter=2000).fit(F(tr, best), ytr)
cut = min(0.0, float(np.percentile([r["lo"]["exist"] for r in tr], 1)))
print(f"tool-exists cut-off (from train): {cut:.2f}")
if "--save" in sys.argv:
    from debussy.paths import HEADS
    HEADS = sys.argv[sys.argv.index("--heads") + 1] if "--heads" in sys.argv else HEADS  # e.g. debussy/heads_k2-horizon.json
    h = json.load(open(HEADS)) if os.path.exists(HEADS) else {}
    h["when2call"] = {"features": best, "coef": m.coef_[0].round(4).tolist(), "intercept": round(float(m.intercept_[0]), 4),
                      "exist_cutoff": round(cut, 4),
                      "note": "P(request_for_info | tool usable); features by 5-fold CV on TRAIN (n=%d); exist_cutoff = 1st pct of TRAIN" % len(tr)}
    json.dump(h, open(HEADS, "w"), indent=1); print("saved when2call head ->", HEADS)
if not te: sys.exit()
y = np.array([r["y"] for r in te]); tcrfi = m.predict(F(te, best))
for name, unavailable in [("A tool-list check", np.array([not r["avail"] for r in te])),
                          ("B model only", np.array([r["lo"]["exist"] < cut for r in te]))]:
    p = np.where(unavailable, 2, tcrfi)
    print(f"{name:18s} acc={np.mean(p == y):.3f}")
    print(confusion_matrix(y, p, labels=[0, 1, 2]))
mask = y != 2
print(f"tool_call vs ask-for-info only (n={mask.sum()}): acc={np.mean(tcrfi[mask] == y[mask]):.3f}")
