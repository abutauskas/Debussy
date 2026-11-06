"""Honesty detector: does reading K2's internal state predict "my answer will be right" better than its confidence?

Trains ONLY on the 400 MMLU-Pro practice questions (K2's last-token hidden state + its top-option probability),
picks the abstain threshold there, then applies both unchanged to the AA public Omniscience stand-in (266 q).
Baseline: K2's own confidence rule (answer only if top probability >= 0.9, also chosen on the practice set).

usage: python scripts/score/honesty_probe.py

Result: the detector scored 51.8 on the stand-in vs 53.1 for the confidence rule (practice, out-of-fold: 62.9 vs 64.9),
and PCA 4/8/16/64 x C 0.01/0.1 variants all stayed below the rule on the practice set too. Debussy keeps the rule.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from debussy.paths import DATA

K = os.path.join(DATA, "k2-horizon")
mm = json.load(open(os.path.join(K, "mmlu_proxy.json"))); om = json.load(open(os.path.join(K, "omni_public.json")))
Em = np.load(os.path.join(K, "emb_mmlu_proxy.npy")); Eo = np.load(os.path.join(K, "emb_omni_public.npy"))
assert len(Em) == len(mm) and len(Eo) == len(om)
Pm = np.array([r["p"] for r in mm]); am = np.array([r["ans"] for r in mm])
Po = np.array([r["p"] for r in om]); ao = np.array([r["ans"] for r in om])


def feats(E, P):
    pmax = P.max(1, keepdims=True); ent = -(P * np.log(P + 1e-12)).sum(1, keepdims=True)
    return np.hstack([E, np.log(pmax / (1 - pmax + 1e-9)), ent])


def component(answer, top, a):
    corr = (answer & (top == a)).mean() * 100; wrong = (answer & (top != a)).sum(); idk = (~answer).sum()
    nh = 100 * (1 - wrong / max(1, wrong + idk)); return corr, nh, (corr + nh) / 2, answer.mean() * 100


Xm, Xo = feats(Em, Pm), feats(Eo, Po); ym = (Pm.argmax(1) == am).astype(int)
probe = lambda C: make_pipeline(StandardScaler(), PCA(n_components=32, random_state=0), LogisticRegression(C=C, max_iter=5000))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
# choose regularisation by cross-validated log-loss on the practice set
best_C = min([0.01, 0.1, 1.0], key=lambda C: -np.mean(
    [np.log(np.clip(p if y else 1 - p, 1e-9, 1)) for p, y in zip(cross_val_predict(probe(C), Xm, ym, cv=cv, method="predict_proba")[:, 1], ym)]))
pc = cross_val_predict(probe(best_C), Xm, ym, cv=cv, method="predict_proba")[:, 1]   # out-of-fold P(correct) on practice set
ths = np.linspace(0.3, 0.95, 27)
t_best = max(ths, key=lambda t: component(pc >= t, Pm.argmax(1), am)[2])
print(f"practice set: {len(ym)} q, K2 right on {ym.mean()*100:.0f}% | probe C={best_C}, abstain threshold chosen = {t_best:.2f}")
print(f"  practice (out-of-fold): confidence rule 0.9 -> {component(Pm.max(1) >= 0.9, Pm.argmax(1), am)[2]:.1f} | "
      f"honesty detector -> {component(pc >= t_best, Pm.argmax(1), am)[2]:.1f}")
m = probe(best_C).fit(Xm, ym); po = m.predict_proba(Xo)[:, 1]
print("\nOmniscience stand-in (266 q, nothing tuned on it):   accuracy  non-halluc  SCORE  answers")
for name, ans in [("K2 confidence rule (>= 0.9)", Po.max(1) >= 0.9), (f"K2 honesty detector (>= {t_best:.2f})", po >= t_best)]:
    c, nh, s, f = component(ans, Po.argmax(1), ao); print(f"  {name:36s} {c:6.1f}    {nh:6.1f}    {s:5.1f}  {f:4.0f}%")
