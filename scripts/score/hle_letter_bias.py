"""HLE letter-bias correction: chosen on an HLE-shaped practice set (MMLU-Pro) only, then applied ONCE to HLE.

A model can lean towards some answer letters whatever the question (e.g. "A", or the last option). The practice set
(transfer_run.py hle_practice: 5-10 options, correct letter uniform, HLE's layout) shows K2's own lean, because there
the right answer is equally likely to be any letter. Candidates:
  none          K2's letter probabilities as they are
  prior         divide by K2's average probability per letter for that option count (uses no answers)
  shared(lam)   one fitted log-bias per letter position + one for "last option", L2 penalty lam
  per_k(lam)    a fitted log-bias per (option count, position), L2 penalty lam
Selection, fixed in advance: nested 5-fold CV accuracy on the practice set (pick inside the inner folds, score on outer
folds never seen); a correction is used only if the selection procedure beats "none" there. HLE is only read for the
final before/after line. Questions with more than 10 options (not covered by the practice set) are left unchanged.

usage: python scripts/score/hle_letter_bias.py [--profile k2-horizon] [--save --heads debussy/heads_k2-horizon.json]
       (--save stores the chosen correction in the heads file, only if the rule says to use one)

Result: K2 leans towards B and away from the last option (E 0.12 vs 0.20 on 5-option questions). Nested CV on the
practice set: 46.7% with the correction vs 45.8% without, so per_k(0.3) is used. HLE, applied once: 13.6% -> 15.0%.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from debussy.paths import DATA

KMAX, NEG = 10, -1e9
K2 = os.path.join(DATA, sys.argv[sys.argv.index("--profile") + 1] if "--profile" in sys.argv else "k2-horizon")


def pad(rows):
    """-> log-probs (n x KMAX, padded with NEG), option counts, answer index."""
    LP = np.full((len(rows), KMAX), NEG); k = np.array([len(r["p"]) for r in rows])
    for i, r in enumerate(rows): LP[i, :k[i]] = np.log(np.clip(r["p"], 1e-9, 1))
    return LP, k, np.array([r["ans"] for r in rows])


def fit_bias(rows, lam, per_k):
    """Maximum likelihood (L2-penalised) letter biases added to K2's log-probs."""
    LP, k, a = pad(rows); n = len(rows); last = np.zeros_like(LP); last[np.arange(n), k - 1] = 1
    valid = LP > NEG / 2; onehot = np.zeros_like(LP); onehot[np.arange(n), a] = 1
    rows_k = (k - 5) if per_k else None; npar = 6 * KMAX if per_k else KMAX + 1

    def bias(th):
        return th.reshape(6, KMAX)[rows_k] if per_k else th[:KMAX][None, :] + th[KMAX] * last

    def f(th):
        S = np.where(valid, LP + bias(th), NEG); S = S - S.max(1, keepdims=True)
        P = np.exp(S) * valid; P /= P.sum(1, keepdims=True)
        nll = -np.sum(np.log(P[np.arange(n), a] + 1e-12)) + lam * th @ th
        G = (P - onehot) * valid
        if per_k:
            g = np.zeros((6, KMAX)); np.add.at(g, rows_k, G); g = g.ravel()
        else:
            g = np.r_[G.sum(0), (G * last).sum()]
        return nll, g + 2 * lam * th

    th = minimize(f, np.zeros(npar), jac=True, method="L-BFGS-B").x
    if per_k:
        B = th.reshape(6, KMAX); fn = lambda r: np.log(np.clip(r["p"], 1e-9, 1)) + B[len(r["p"]) - 5, :len(r["p"])]
        fn.params = {"k_min": 5, "k_max": KMAX, "lam": lam, "bias": [[round(float(v), 5) for v in row[:5 + i]] for i, row in enumerate(B)]}
        return fn
    b, c = th[:KMAX], th[KMAX]
    return lambda r: np.log(np.clip(r["p"], 1e-9, 1)) + b[:len(r["p"])] + c * (np.arange(len(r["p"])) == len(r["p"]) - 1)


def fit_prior(rows):
    pri = {kk: np.mean([r["p"] for r in rows if len(r["p"]) == kk], 0) for kk in {len(r["p"]) for r in rows}}
    return lambda r: np.log(np.clip(r["p"], 1e-9, 1)) - np.log(pri[len(r["p"])])


CANDIDATES = {"none": lambda rows: (lambda r: np.log(np.clip(r["p"], 1e-9, 1))), "prior": fit_prior,
              **{f"shared({l})": (lambda l: lambda rows: fit_bias(rows, l, False))(l) for l in (0.3, 3.0)},
              **{f"per_k({l})": (lambda l: lambda rows: fit_bias(rows, l, True))(l) for l in (0.3, 3.0)}}


def acc(score, rows): return float(np.mean([int(np.argmax(score(r)) == r["ans"]) for r in rows]))


def cv_acc(name, rows, seed=0):
    return np.mean([acc(CANDIDATES[name]([rows[i] for i in a]), [rows[i] for i in b])
                    for a, b in KFold(5, shuffle=True, random_state=seed).split(rows)])


def pick(rows):
    cv = {m: cv_acc(m, rows) for m in CANDIDATES}; m = max(cv, key=cv.get); return m, cv


if __name__ == "__main__":
    pr = json.load(open(os.path.join(K2, "hle_practice.json")))
    print(f"practice set: {len(pr)} MMLU-Pro questions, 5-10 options; K2 plain accuracy {acc(CANDIDATES['none'](pr), pr)*100:.1f}%")
    # K2's lean on the practice set: average probability per letter vs uniform, for 5-option questions
    p5 = np.array([r["p"] for r in pr if len(r["p"]) == 5]); print("  5-option questions, K2's average probability per letter:",
                                                                   " ".join(f"{chr(65+i)}={v:.2f}" for i, v in enumerate(p5.mean(0))), "(uniform 0.20)")
    best, cv = pick(pr)
    print("  inner CV accuracy: " + "  ".join(f"{m}={v*100:.1f}" for m, v in cv.items()))
    nested, plain = [], []
    for a, b in KFold(5, shuffle=True, random_state=1).split(pr):
        ra, rb = [pr[i] for i in a], [pr[i] for i in b]
        m, _ = pick(ra); nested.append(acc(CANDIDATES[m](ra), rb)); plain.append(acc(CANDIDATES["none"](ra), rb))
    use = np.mean(nested) > np.mean(plain)
    print(f"  nested CV: selection procedure {np.mean(nested)*100:.1f}% vs none {np.mean(plain)*100:.1f}% -> "
          + (f"use '{best}'" if use else "keep 'none'"))
    # the single look at HLE
    hl = json.load(open(os.path.join(K2, "hle.json")))
    rows = [{"p": r["p"], "ans": r["letters"].index(r["ans"])} for r in hl]
    corr = CANDIDATES[best](pr) if use else CANDIDATES["none"](pr)
    covered = [r for r in rows if 5 <= len(r["p"]) <= KMAX]
    after = np.mean([int(np.argmax(corr(r) if 5 <= len(r["p"]) <= KMAX else r["p"]) == r["ans"]) for r in rows])
    before = np.mean([int(np.argmax(r["p"]) == r["ans"]) for r in rows])
    if "--save" in sys.argv and use and hasattr(corr, "params"):
        hp = sys.argv[sys.argv.index("--heads") + 1]; h = json.load(open(hp))
        h["letter_bias"] = {"method": best, "chosen_on": f"{len(pr)} HLE-shaped MMLU-Pro practice questions (nested CV)", **corr.params}
        json.dump(h, open(hp, "w"), indent=1); print("saved letter_bias ->", hp)
    print(f"\nHLE-MultiChoice ({len(rows)} q; {len(covered)} with 5-10 options get the correction): "
          f"before {before*100:.1f}%  after {after*100:.1f}%  (chance {np.mean([1/len(r['p']) for r in rows])*100:.1f}%)")
