"""Compare two base models on IDENTICAL calibration/training and test rows (WildGuard, When2Call).

usage: python scripts/score/compare_bases.py wg     |   python scripts/score/compare_bases.py w2c
Uses data/k2-horizon/{wg,w2c}.json (from transfer_run.py) and the matching rows of Qwen's saved results.
All choices (head type, threshold, features) are made on the calibration/training rows only.
"""
import os, sys, json, subprocess, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
from debussy.paths import ROOT

os.chdir(ROOT)
which = sys.argv[1]
K2 = json.load(open(f"data/k2-horizon/{which}.json"))
ktr = [r for r in K2 if r["split"] == "tr"]; kte = [r for r in K2 if r["split"] == "te"]

if which == "wg":
    Q = json.load(open("data/wg_all_718tr_880te.json"))   # train order: same rows as all_rows() first; test: same order
    qtr = [r for r in Q if r["split"] == "tr"][:len(ktr)]; qte = [r for r in Q if r["split"] == "te"][:len(kte)]
    # add the natural-mix calibration rows, when K2 has them (same rows as Qwen's wg_natural_train*.json)
    for b, qfile in [(1, "data/wg_natural_train.json"), (2, "data/wg_natural_train2.json")]:
        kf = f"data/k2-horizon/wg_natural_b{b}.json"
        if os.path.exists(kf):
            kn = json.load(open(kf)); qn = json.load(open(qfile))[:len(kn)]
            ktr += kn; qtr += qn
    assert [r["y"] for r in qtr] == [r["y"] for r in ktr] and [r["y"] for r in qte] == [r["y"] for r in kte], "row mismatch"
    scorer, extra = "scripts/score/wildguard.py", ["--natural-weights"]
else:
    from debussy.skills.when2call import build_items
    Q = json.load(open("data/w2o2_4b_799tr_600te.json"))
    q_tr_rows = [r for r in Q if r["split"] == "tr"][:300]; q_tr_items = [it for it in build_items(150, 0) if it["split"] == "tr"]
    key = {(it["q"], it["call"]): r for it, r in zip(q_tr_items, q_tr_rows)}
    n_per = len(ktr) // 2
    k_tr_items = [it for it in build_items(n_per, 0) if it["split"] == "tr"]
    qtr = [key[(it["q"], it["call"])] for it in k_tr_items]
    qte = [r for r in Q if r["split"] == "te"][:len(kte)]
    assert [r["y"] for r in qtr] == [r["y"] for r in ktr] and [r["y"] for r in qte] == [r["y"] for r in kte], "row mismatch"
    scorer, extra = "scripts/score/when2call.py", []

print(f"{which}: {len(ktr)} calibration/training rows, {len(kte)} test rows, identical for both models\n")
for name, rows in [("Qwen3.5-4B", qtr + qte), ("K2-Horizon-7B", ktr + kte)]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(rows, f)
    out = subprocess.run([sys.executable, scorer, f.name] + extra, capture_output=True, text=True).stdout
    print(f"=== {name}"); print("\n".join(l for l in out.splitlines() if not l.startswith("  train-CV")))
