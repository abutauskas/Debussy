"""Extract frozen-LM hidden states at the decision point for When2Call train/test (resumable).

usage: python extract_lm_feats.py MODEL_ID N_TRAIN_PER_CLASS N_TEST OUT.npz
"""
import json, os, random, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_decide import build_prompt, LABELS, D
from when2call_head import label_pref

torch.set_num_threads(4)
mid, ntr, nte, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

P = [json.loads(l) for l in open(D + "w2c_when2call_train_pref.jsonl")]
random.Random(1).shuffle(P)
train, per = [], {0: 0, 1: 0, 2: 0}
for p in P:
    y = label_pref(p["chosen_response"]["content"])
    if y is not None and per[y] < ntr:
        per[y] += 1; train.append((p["messages"][-1]["content"], p["tools"], y))
T = [json.loads(l) for l in open(D + "w2c_when2call_test_mcq.jsonl")]
random.Random(0).shuffle(T)
test = [(t["question"], t["tools"], LABELS.index(t["correct_answer"])) for t in T[:nte]]
items = [("tr", *x) for x in train] + [("te", *x) for x in test]

tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).eval()
L = model.config.get_text_config().num_hidden_layers if hasattr(model.config, "get_text_config") else model.config.num_hidden_layers
layers = sorted({L // 2, (3 * L) // 4, L})

part = out + ".partial.npz"
done = dict(np.load(part, allow_pickle=True)) if os.path.exists(part) else {}
feats = list(done.get("X", [])); start = len(feats)
t0 = time.time()
for i in range(start, len(items)):
    _, q, tools, y = items[i]
    x = tok(build_prompt(tok, q, tools), return_tensors="pt", truncation=True, max_length=2048)
    with torch.inference_mode():
        hs = model(**x, output_hidden_states=True).hidden_states
    feats.append(np.concatenate([hs[l][0, -1].float().numpy() for l in layers]))
    if (i + 1) % 100 == 0 or i + 1 == len(items):
        np.savez(part, X=np.array(feats))
        print(f"{i+1}/{len(items)} {(time.time()-t0)/(i+1-start):.2f}s/ex", flush=True)

X = np.array(feats); split = np.array([s for s, *_ in items]); y = np.array([it[3] for it in items])
np.savez_compressed(out, Xtr=X[split == "tr"], ytr=y[split == "tr"], Xte=X[split == "te"], yte=y[split == "te"], layers=layers)
print("DONE", X.shape, layers)
