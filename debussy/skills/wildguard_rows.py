"""Rebuild the exact WildGuard rows behind data/wg1p_4b_all_468tr.json, in the same order.

Order: 120 calibration (wildguard_yesno) + 178 (wg_more_train seed 1) + 170 (seed 2) + 300 test + 200 test.
"""
import os
import pandas as pd
from ..paths import ROOT as _ROOT

D = _ROOT + "/data/"
KEYS = ["prompt_harm_label", "response_refusal_label", "response_harm_label"]


def all_rows():
    tr = pd.read_parquet(D + "wg_wildguard_train.parquet").dropna(subset=["response_harm_label", "response_refusal_label"])
    idx0 = [i for _, g in tr.groupby(KEYS) for i in g.sample(min(len(g), 120 // 6 + 1), random_state=0).index]
    cal0 = tr.loc[idx0].sample(frac=1, random_state=0).head(120)
    rest = tr.drop(index=list(cal0.index)); batches = []
    for s in (1, 2):
        idx = [i for _, g in rest.groupby(KEYS) for i in g.sample(min(len(g), 200 // 6 + 1), random_state=s).index]
        b = rest.loc[idx].sample(frac=1, random_state=s).head(200); batches.append(b); rest = rest.drop(index=list(b.index))
    te = pd.read_parquet(D + "wg_wildguard_test.parquet")
    te = te[(te.prompt_harm_agreement == 3) & (te.response_refusal_agreement == 3) & (te.response_harm_agreement == 3)]
    te500 = te.sample(500, random_state=0)
    parts = [("tr", cal0), ("tr", batches[0]), ("tr", batches[1]), ("te", te500)]
    return [(s, r) for s, df in parts for _, r in df.iterrows()]


def natural_rows(n, batch=1):
    """Calibration rows drawn at random (the training split's natural label mix), excluding every row in all_rows().
    Batch 1 = 170 rows (seed 7); batch b > 1 first removes the earlier batches (170 rows each), then samples with seed 6+b.
    Identical to what scripts/run/wildguard_natural_train.py used for Qwen, so both base models see the same rows."""
    tr = pd.read_parquet(D + "wg_wildguard_train.parquet").dropna(subset=["response_harm_label", "response_refusal_label"])
    used = {(r["prompt"], r["response"]) for s, r in all_rows() if s == "tr"}
    pool = tr[~pd.Series([(p, q) in used for p, q in zip(tr.prompt, tr.response)], index=tr.index)]
    for b in range(1, batch):
        pool = pool.drop(index=list(pool.sample(170, random_state=6 + b).index))
    return pool.sample(n, random_state=6 + batch)
