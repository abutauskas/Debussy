"""Collect K2-Horizon's final hidden state at the harmful-reply decision point, for the WildGuard probe.

Needs a llama-server started with: --embeddings --pooling last --embd-normalize -1
For each WildGuard row it sends exactly the text the chained single pass has seen when it answers Q3 (harmful reply):
the System 1 prompt, then K2's OWN Q1/Q2 answers (the signs of its saved log-odds), then "Q3:".
Rows and order match K2's combined file from `compare_bases.py wg --write`: calibration = all_rows() calibration rows,
then natural-mix batches 1 and 2 (718); test = all_rows() test rows (500). Labels are checked row by row.

usage: python scripts/run/k2_wg_embed.py cal|te    (resumable; writes data/k2-horizon/emb_wg_<cal|te>.npy)
       DEADLINE=<unix seconds> stops cleanly before that time, so a bounded session can end and the next resume.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
import json, time
import numpy as np
import pandas as pd
from debussy.engine import PROFILES, _Server
from debussy.paths import DATA
from debussy.skills.wildguard import QUESTIONS, COL, POS, chain_text
from debussy.skills.wildguard_rows import all_rows, natural_rows

P = PROFILES["k2-horizon"]; URL = os.environ.get("DEBUSSY_SERVER", "http://127.0.0.1:8080"); b = _Server(URL)
K = os.path.join(DATA, "k2-horizon")


def clip(s, k):  # identical to transfer_run.py's server-side clip
    t = b.tok(str(s))
    return str(s) if len(t) <= k else b.detok(t[:k]) + " [...]"


def q3_prompt(row, lo):  # the server adds BOS itself, as in Engine on the server backend
    yn = lambda v: " yes" if v > 0 else " no"
    return (P["user_start"] + chain_text(row["prompt"], row["response"], clip) + P["assistant_start"]
            + f"Q1:{yn(lo['prompt_harm'])}\nQ2:{yn(lo['refusal'])}\nQ3:")


def embed(text):
    r = b._post("/embedding", {"content": text})
    e = np.array(r[0]["embedding"] if isinstance(r, list) else r["embedding"], dtype=np.float32)
    return e[-1] if e.ndim == 2 else e


if __name__ == "__main__":
    part = sys.argv[1]; assert part in ("cal", "te") and list(QUESTIONS) == ["prompt_harm", "refusal", "response_harm"]
    saved = json.load(open(os.path.join(K, "wg_all_718cal_500te.json")))
    R = all_rows()
    if part == "cal":
        rows = [r for s, r in R if s == "tr"] + [r for _, r in natural_rows(170, 1).iterrows()] + [r for _, r in natural_rows(80, 2).iterrows()]
        res = [x for x in saved if x["split"] == "tr"]
    else:
        rows = [r for s, r in R if s == "te"]; res = [x for x in saved if x["split"] == "te"]
    assert len(rows) == len(res), (len(rows), len(res))
    for r, x in zip(rows, res):  # same rows, same order
        assert x["y"] == {k: (None if pd.isna(r[c]) else int(r[c] == POS[k])) for k, c in COL.items()}
    out = os.path.join(K, f"emb_wg_{part}.npy")
    have = list(np.load(out)) if os.path.exists(out) else []
    deadline = float(os.environ.get("DEADLINE", "inf")); t0 = time.time(); s0 = len(have)
    for i in range(s0, len(rows)):
        if time.time() > deadline - 120:
            np.save(out, np.array(have)); print(f"STOPPED at deadline: {part} {len(have)}/{len(rows)} saved", flush=True); sys.exit(0)
        have.append(embed(q3_prompt(rows[i], res[i]["lo"])))
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            np.save(out, np.array(have)); print(f"emb_wg {part} {i+1}/{len(rows)} {(time.time()-t0)/(i+1-s0):.2f}s/row", flush=True)
    print("DONE", part, np.array(have).shape)
