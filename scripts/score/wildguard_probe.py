"""WildGuard hidden-state probe on K2: does reading K2's internal state at the harmful-reply decision point ("Q3:")
beat the current calibrated heads?

Probe = logistic regression on PCA(K2 hidden state) plus the same log-odds features the heads use (3 log-odds and
3 interactions), with an F1-tuned threshold. Everything is chosen on the 718 calibration rows only: the same 5 folds,
natural label-mix weights and F1 rule as scripts/score/wildguard.py. Rule fixed in advance: a probe replaces the
current head for a question only if it wins on NESTED calibration CV (the whole pick-the-best-variant procedure is
scored on folds it never saw, for the probe and for the heads alike). A plain CV comparison is not enough: with random
numbers in place of hidden states, the best of 12 probe variants still "won" plain CV on all three questions and then
lost on test. The 500 test rows are only used to report both.

Data: data/k2-horizon/wg_all_718cal_500te.json (compare_bases.py wg --write) and emb_wg_{cal,te}.npy (k2_wg_embed.py).
usage: python scripts/score/wildguard_probe.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                           # scripts/score (wildguard.py)
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from debussy.paths import DATA
import wildguard as wg

wg.NATURAL = True
K2 = os.path.join(DATA, "k2-horizon")
rows = json.load(open(os.path.join(K2, "wg_all_718cal_500te.json")))
tr = [r for r in rows if r["split"] == "tr"]; te = [r for r in rows if r["split"] == "te"]
Etr, Ete = np.load(os.path.join(K2, "emb_wg_cal.npy")), np.load(os.path.join(K2, "emb_wg_te.npy"))
assert len(Etr) == len(tr) and len(Ete) == len(te), (len(Etr), len(tr), len(Ete), len(te))
VARIANTS = [(k, C) for k in (8, 16, 32, 64) for C in (0.01, 0.1, 1.0)]
FOLDS = lambda y, seed=0: StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def probe_fit(E, rs, y, w, k, C):
    se, sx = StandardScaler().fit(E), StandardScaler().fit(wg.X(rs, True))
    pca = PCA(n_components=k, random_state=0).fit(se.transform(E))
    Z = lambda E_, rs_: np.c_[pca.transform(se.transform(E_)), sx.transform(wg.X(rs_, True))]
    m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Z(E, rs), y, sample_weight=w)
    p = m.predict_proba(Z(E, rs))[:, 1]
    th = max(np.linspace(0.1, 0.9, 33), key=lambda t: wg.f1w(y, (p > t).astype(int), w))
    return lambda E_, rs_: (m.predict_proba(Z(E_, rs_))[:, 1] > th).astype(int)


def probe_cv(E, rs, y, w, k, C):
    return np.mean([wg.f1w(y[b], probe_fit(E[a], [rs[i] for i in a], y[a], w[a], k, C)(E[b], [rs[i] for i in b]), w[b])
                    for a, b in FOLDS(y)])


def pick_head(rs, y, w, j):
    cv = {m: wg.cv_f1(m, rs, y, w, j) for m in wg.METHODS}; m = max(cv, key=cv.get); return m, cv[m]


def pick_probe(E, rs, y, w):
    cv = {v: probe_cv(E, rs, y, w, *v) for v in VARIANTS}; v = max(cv, key=cv.get); return v, cv[v]


def nested(E, rs, y, w, j):
    """Outer 5 folds (seed 1): pick on the outer-train part only, score on the held-out part. -> (head, probe) F1."""
    h, p = [], []
    for a, b in FOLDS(y, 1):
        ra, rb = [rs[i] for i in a], [rs[i] for i in b]
        m, _ = pick_head(ra, y[a], w[a], j); h.append(wg.f1w(y[b], wg.fit(m, ra, y[a], w[a], j)[0](rb), w[b]))
        v, _ = pick_probe(E[a], ra, y[a], w[a]); p.append(wg.f1w(y[b], probe_fit(E[a], ra, y[a], w[a], *v)(E[b], rb), w[b]))
    return np.mean(h), np.mean(p)


print(f"calibration {len(tr)} rows, test {len(te)} rows; probe variants (PCA k, C): {len(VARIANTS)}\n")
print(f"{'question':14s} | {'current head':8s} {'nested':>6s} {'test':>6s} | {'best probe':18s} {'nested':>6s} {'test':>6s} | used")
final, base = [], []
for j, q in enumerate(wg.K):
    itr = [i for i, r in enumerate(tr) if r["y"][q] is not None]; ite = [i for i, r in enumerate(te) if r["y"][q] is not None]
    trk, tek = [tr[i] for i in itr], [te[i] for i in ite]
    ytr = np.array([r["y"][q] for r in trk]); yte = np.array([r["y"][q] for r in tek]); w = wg.natural_weights(trk)
    Ek, Etk = Etr[itr], Ete[ite]
    mb, _ = pick_head(trk, ytr, w, j); tb = f1_score(yte, wg.fit(mb, trk, ytr, w, j)[0](tek))
    vb, _ = pick_probe(Ek, trk, ytr, w); tp = f1_score(yte, probe_fit(Ek, trk, ytr, w, *vb)(Etk, tek))
    nh, npr = nested(Ek, trk, ytr, w, j)
    use_probe = npr > nh; final.append(tp if use_probe else tb); base.append(tb)
    print(f"{q:14s} | {mb:8s} {nh:6.3f} {tb:6.3f} | PCA {vb[0]:<3d} C={vb[1]:<8g} {npr:6.3f} {tp:6.3f} | "
          + ("probe" if use_probe else "head"), flush=True)
print(f"\nAVERAGE F1 on test: current heads {np.mean(base):.3f} | with the probe where it won on calibration {np.mean(final):.3f}")
