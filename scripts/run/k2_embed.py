"""Collect K2-Horizon's final hidden state ("what it is thinking") at the answer point, for the honesty detector.

Needs a llama-server started with: --embeddings --pooling last --embd-normalize -1
For each knowledge question it sends exactly the prompt used for answering (System 1 format, ending at "Answer:")
and stores the last-token hidden state. Rows line up 1:1 with data/k2-horizon/mmlu_proxy.json and omni_public.json.

usage: python scripts/run/k2_embed.py mmlu_proxy|omni_public     (resumable; writes data/k2-horizon/emb_<name>.npy)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
import json, random, time, urllib.request
import numpy as np
from debussy.engine import PROFILES
from debussy.paths import DATA

P = PROFILES["k2-horizon"]; URL = os.environ.get("DEBUSSY_SERVER", "http://127.0.0.1:8080")
MC = "Answer the multiple-choice question with the letter of the correct option.\n\n"; L4 = ["A", "B", "C", "D"]


def prompt(user_text):  # identical to Engine.letter_probs on the server backend (the server adds BOS itself)
    return P["user_start"] + user_text + P["assistant_start"] + P["answer_cue"]


def embed(text):
    req = urllib.request.Request(URL + "/embedding", data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    e = r[0]["embedding"] if isinstance(r, list) else r["embedding"]
    e = np.array(e, dtype=np.float32)
    return e[-1] if e.ndim == 2 else e


def items(name):
    if name == "mmlu_proxy":
        from debussy.skills.knowledge_proxy import build
        return [MC + q["q"] + "\n\n" + "\n".join(f"{L4[k]}. {o}" for k, o in enumerate(q["opts"])) for q in build(400)]
    xs = [x for x in json.load(open(os.path.join(DATA, "omni_public_gen.json"))) if len(x["wrong"]) == 3]
    out = []
    for x in xs:
        opts = x["wrong"] + [x["answer"]]; random.Random(x["qid"]).shuffle(opts)
        out.append(MC + x["q"] + "\n\n" + "\n".join(f"{L4[k]}. {o}" for k, o in enumerate(opts)))
    return out


if __name__ == "__main__":
    name = sys.argv[1]; texts = items(name)
    out = os.path.join(DATA, "k2-horizon", f"emb_{name}.npy")
    have = list(np.load(out)) if os.path.exists(out) else []
    t0 = time.time(); s0 = len(have)
    for i in range(s0, len(texts)):
        have.append(embed(prompt(texts[i])))
        if (i + 1) % 50 == 0 or i + 1 == len(texts):
            np.save(out, np.array(have)); print(f"emb {name} {i+1}/{len(texts)} {(time.time()-t0)/(i+1-s0):.2f}s/q", flush=True)
    print("DONE", name, np.array(have).shape)
