"""When2Call decision head: tiny frozen embedder + small trained classifier.

Decides, for a user request plus a list of available tools, whether to:
  tool_call | request_for_info | cannot_answer
Trained on When2Call train_pref (9k), scored on the full When2Call test MCQ (3,652).
"""
import os
import json, re, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

torch.set_num_threads(4)
EMB = "BAAI/bge-small-en-v1.5"
D = _ROOT + "/data/"
LABELS = ["tool_call", "request_for_info", "cannot_answer"]

tok = AutoTokenizer.from_pretrained(EMB)
enc = AutoModel.from_pretrained(EMB).eval()


@torch.inference_mode()
def embed(texts, bs=64, maxlen=256):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        h = enc(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def tool_text(t):
    t = json.loads(t) if isinstance(t, str) else t
    p = t.get("parameters", {}) or {}
    props = p.get("properties", {}) or {}
    req = p.get("required", t.get("required", [])) or []
    return t.get("name", ""), t.get("description", ""), props, req


def label_pref(t):
    if "<TOOLCALL>" in t: return 0
    s = t.strip()
    if re.search(r"\b(unable|can't|cannot|can not|don't have|do not have|not able|isn't possible|beyond my)\b", s, re.I) and not s.endswith("?"):
        return 2
    if "?" in s: return 1
    return None


def surface(q, props, req):
    """Cheap lexical signals: are the required argument values present in the request?"""
    ql = q.lower()
    nums = len(re.findall(r"\d", q))
    hit = 0
    for r in req:
        words = [w for w in re.split(r"[_\W]+", r.lower()) if len(w) > 2]
        if any(w in ql for w in words): hit += 1
    no_default = sum(1 for r in req if not str((props.get(r) or {}).get("default", "")).strip())
    return [len(req), no_default, hit / max(1, len(req)), min(nums, 20) / 20, len(q) / 300,
            q.count('"') / 4, int("?" in q), len(re.findall(r"\b[A-Z][a-z]+", q)) / 10]


def featurize(questions, toolsets):
    qe = embed(questions)
    flat, owner = [], []
    for i, ts in enumerate(toolsets):
        for t in ts:
            n, d, props, req = tool_text(t)
            flat.append(f"{n}: {d} Required: {', '.join(req)}")
            owner.append(i)
    te = embed(flat)
    feats = []
    for i, q in enumerate(questions):
        idx = [k for k, o in enumerate(owner) if o == i]
        if not idx:
            best, sims, s = np.zeros(384, np.float32), np.array([0.0]), [0] * 8
            nt = 0
        else:
            sims = te[idx] @ qe[i]
            j = idx[int(sims.argmax())]
            best = te[j]
            _, _, props, req = tool_text(toolsets[i][int(sims.argmax())])
            s = surface(q, props, req)
            nt = len(idx)
        feats.append(np.concatenate([qe[i], best, qe[i] * best,
                                     [sims.max(), sims.mean(), nt / 10, float(nt == 0)], s]))
    return np.array(feats, np.float32)


if __name__ == "__main__":
    t0 = time.time()
    P = [json.loads(l) for l in open(D + "w2c_when2call_train_pref.jsonl")]
    tr = [(p["messages"][-1]["content"], p["tools"], label_pref(p["chosen_response"]["content"])) for p in P]
    tr = [x for x in tr if x[2] is not None]
    T = [json.loads(l) for l in open(D + "w2c_when2call_test_mcq.jsonl")]
    Xtr = featurize([x[0] for x in tr], [x[1] for x in tr]); ytr = np.array([x[2] for x in tr])
    Xte = featurize([t["question"] for t in T], [t["tools"] for t in T])
    yte = np.array([LABELS.index(t["correct_answer"]) for t in T])
    np.savez_compressed(D + "w2c_feats.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"features {Xtr.shape} {Xte.shape} in {time.time()-t0:.0f}s", flush=True)
