"""Compare two base models on IDENTICAL calibration/training and test rows (WildGuard, When2Call).

usage: python scripts/score/compare_bases.py wg [--write]   |   python scripts/score/compare_bases.py w2c
Uses data/k2-horizon/{wg,w2c}.json (from transfer_run.py), plus any wg_tr_*/wg_te_* extensions and natural-mix batches,
and the matching rows of Qwen's saved results. --write also saves K2's combined rows to one file.
All choices (head type, threshold, features) are made on the calibration/training rows only.
"""
import os, sys, json, subprocess, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
from debussy.paths import ROOT

os.chdir(ROOT)
which = sys.argv[1]
K2 = json.load(open(f"data/k2-horizon/{which}.json"))
ktr = [r for r in K2 if r["split"] == "tr"]; kte = [r for r in K2 if r["split"] == "te"]


def extend(rows, split):
    """Append later rows of all_rows() saved by `transfer_run.py wg_more_cal|wg_more_test` (wg_tr_A_B.json, wg_te_A_B.json)."""
    import glob
    while True:
        files = sorted(glob.glob(f"data/k2-horizon/wg_{split}_{len(rows)}_*.json"))
        if not files: return rows
        start, end = len(rows), int(files[-1].rsplit("_", 1)[1][:-5]); more = json.load(open(files[-1]))
        rows = rows + more
        if len(more) < end - start:  # still running: use what is saved, and stop here
            print(f"note: {files[-1]} is unfinished ({len(more)}/{end - start} rows)"); return rows


if which == "wg":
    # the first 150 calibration rows in wg.json differ from a fresh re-run (by ~0.4 log-odds on the 2 we checked; most
    # likely saved before the double-BOS fix), so use the re-run (transfer_run.py wg_more_cal 0 150) once it is complete.
    # The test rows match a fresh run exactly (parity check: 0.0000).
    fresh = f"data/k2-horizon/wg_tr_0_{len(ktr)}.json"
    if os.path.exists(fresh) and len(json.load(open(fresh))) == len(ktr):
        ktr = json.load(open(fresh)); print(f"using re-run calibration rows {fresh}")
    ktr = extend(ktr, "tr"); kte = extend(kte, "te")
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
if "--write" in sys.argv:  # K2's combined rows, for: python scripts/score/wildguard.py FILE --natural-weights --save --heads ...
    out = f"data/k2-horizon/{which}_all_{len(ktr)}cal_{len(kte)}te.json"; json.dump(ktr + kte, open(out, "w")); print("wrote", out, "\n")
for name, rows in [("Qwen3.5-4B", qtr + qte), ("K2-Horizon-7B", ktr + kte)]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(rows, f)
    out = subprocess.run([sys.executable, scorer, f.name] + extra, capture_output=True, text=True).stdout
    print(f"=== {name}"); print("\n".join(l for l in out.splitlines() if not l.startswith("  train-CV")))
