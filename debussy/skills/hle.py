"""HLE-MultiChoice, Artificial Analysis style: 513 text-only multiple-choice questions, one choice among the
question's lettered answers (original order), scored by exact match on the letter. One pass per question:
Debussy's probability for each answer letter; prediction = most likely letter. Resumable.

usage: python -I hle_eval.py OUT.json            (then: python -I hle_eval.py --score OUT.json)
"""
import json, os, re, sys, time
import numpy as np

from ..paths import ROOT as HERE


def load():
    import pandas as pd
    d = pd.read_parquet(os.path.join(HERE, "data/hle/hle_test.parquet"),
                        columns=["id", "question", "image", "answer", "answer_type", "category"])
    img = d.image.fillna("").astype(str).str.len() > 0
    d = d[(d.answer_type == "multipleChoice") & (~img)]
    out = []
    for _, r in d.iterrows():
        letters = list(dict.fromkeys(re.findall(r"(?m)^\s*\(?([A-Z])[\.\)]\s", r.question)))
        out.append({"id": r.id, "q": r.question, "letters": letters, "ans": r.answer.strip(), "cat": r.category})
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--score":
        rows = json.load(open(sys.argv[2]))
        acc = np.mean([r["pred"] == r["ans"] for r in rows]) * 100
        chance = np.mean([100 / len(r["letters"]) for r in rows])
        print(f"HLE-MultiChoice: {len(rows)} questions | accuracy {acc:.1f}% | chance level {chance:.1f}%")
        cats = sorted(set(r["cat"] for r in rows))
        for c in cats:
            rs = [r for r in rows if r["cat"] == c]
            print(f"  {c:28s} n={len(rs):3d}  acc {np.mean([r['pred'] == r['ans'] for r in rs]) * 100:5.1f}%"
                  f"  chance {np.mean([100 / len(r['letters']) for r in rs]):5.1f}%")
        sys.exit()
    import llama_cpp
    from llama_cpp import Llama
    out = sys.argv[1]; qs = load()
    llm = Llama(model_path=os.path.join(HERE, "models/Qwen3.5-4B-Q4_K_M.gguf"), n_ctx=8192, n_threads=4, n_batch=512, verbose=False)
    rows = json.load(open(out)) if os.path.exists(out) else []
    t0 = time.time(); s0 = len(rows)
    for i in range(s0, len(qs)):
        q = qs[i]
        p = ("<|im_start|>user\nAnswer the multiple-choice question with the letter of the correct option.\n\n"
             f"{q['q']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ")
        toks = llm.tokenize(p.encode(), add_bos=False, special=True)
        llm.reset(); llm.eval(toks[-8000:])
        ids = [llm.tokenize(l.encode(), add_bos=False)[0] for l in q["letters"]]
        lg = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1), shape=(llm.n_vocab(),)).astype(np.float64)[ids]
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        rows.append({"id": q["id"], "letters": q["letters"], "p": pr.round(5).tolist(), "pred": q["letters"][int(pr.argmax())],
                     "ans": q["ans"], "cat": q["cat"], "n_tokens": len(toks)})
        if (i + 1) % 25 == 0 or i + 1 == len(qs):
            json.dump(rows, open(out, "w")); print(f"{i+1}/{len(qs)} {(time.time()-t0)/(i+1-s0):.2f}s/q", flush=True)
    print("DONE")
