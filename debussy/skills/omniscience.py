"""Closer stand-in for AA-Omniscience-Decision, built from Artificial Analysis's official public sample
(ArtificialAnalysis/AA-Omniscience-Public-v1.1: 300 open-answer questions, Apache-2.0).

The Decision version shows 4 answers (1 correct) + "I don't know". The public sample has no wrong answers, so:
  phase 1 (gen):   the model writes 3 plausible but wrong answers for each question (it is shown the correct answer
                   so it can avoid it). Fixed seed shuffle into A-D.
  phase 2 (eval):  a separate pass, without the correct answer, gives Debussy's probability for each letter.
  --score:         applies Debussy's ALREADY-CHOSEN rule (answer only if top probability >= heads.json idk_threshold,
                   else "I don't know") and reports AA's metrics. Nothing is tuned on these questions.
Caveat: distractors written by the same model may be easier or harder than Artificial Analysis's own.

usage: python omni_public.py gen OUT_GEN.json | eval OUT_GEN.json OUT_EVAL.json | --score OUT_EVAL.json
"""
import json, os, random, re, sys, time
import numpy as np

from ..paths import ROOT as HERE, HEADS
MODEL = os.path.join(HERE, "models/Qwen3.5-4B-Q4_K_M.gguf")
L = ["A", "B", "C", "D"]


def llm(n_ctx=2048):
    from llama_cpp import Llama
    return Llama(model_path=MODEL, n_ctx=n_ctx, n_threads=4, n_batch=512, verbose=False)


def chat(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def norm(s): return re.sub(r"\W+", " ", str(s).lower()).strip()


def gen(out):
    import pandas as pd
    d = pd.read_csv(os.path.join(HERE, "data/omni_public/omni_public_v1_1.csv"))
    rows = json.load(open(out)) if os.path.exists(out) else []
    m = llm(); t0 = time.time(); s0 = len(rows)
    for i in range(s0, len(d)):
        r = d.iloc[i]
        p = chat("Write exactly three different answers to the question below that sound plausible to an expert but are "
                 "WRONG. Match the style, length and units of the correct answer. One answer per line, no numbering, no "
                 f"explanations.\n\nQuestion: {r.question}\nCorrect answer (do NOT repeat it): {r.answer}")
        txt = m.create_completion(p, max_tokens=120, temperature=0.0, stop=["<|im_end|>"])["choices"][0]["text"]
        cands = [re.sub(r"^\s*(?:[-*•]|\d+[.)]|[A-D][.)])\s*", "", l).strip() for l in txt.splitlines()]
        wrong = []
        for c in cands:
            if c and norm(c) != norm(r.answer) and norm(c) not in {norm(w) for w in wrong}:
                wrong.append(c)
        rows.append({"qid": int(r.question_id), "domain": r.domain, "q": r.question, "answer": str(r.answer), "wrong": wrong[:3]})
        if (i + 1) % 25 == 0 or i + 1 == len(d):
            json.dump(rows, open(out, "w")); print(f"gen {i+1}/{len(d)} {(time.time()-t0)/(i+1-s0):.2f}s/q", flush=True)
    print("DONE gen")


def evaluate(src, out):
    import llama_cpp
    items = [x for x in json.load(open(src)) if len(x["wrong"]) == 3]
    rows = json.load(open(out)) if os.path.exists(out) else []
    m = llm(); ids = [m.tokenize(l.encode(), add_bos=False)[0] for l in L]; t0 = time.time(); s0 = len(rows)
    for i in range(s0, len(items)):
        x = items[i]; opts = x["wrong"] + [x["answer"]]; random.Random(x["qid"]).shuffle(opts)
        body = x["q"] + "\n\n" + "\n".join(f"{L[k]}. {o}" for k, o in enumerate(opts))
        m.reset(); m.eval(m.tokenize((chat("Answer the multiple-choice question with the letter of the correct option.\n\n" + body)
                                      + "Answer: ").encode(), add_bos=False, special=True))
        lg = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(m._ctx.ctx, -1), shape=(m.n_vocab(),)).astype(np.float64)[ids]
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        rows.append({"qid": x["qid"], "domain": x["domain"], "p": pr.round(5).tolist(), "ans": opts.index(x["answer"])})
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            json.dump(rows, open(out, "w")); print(f"eval {i+1}/{len(items)} {(time.time()-t0)/(i+1-s0):.2f}s/q", flush=True)
    print("DONE eval")


def score(src):
    rows = json.load(open(src)); P = np.array([r["p"] for r in rows]); ans = np.array([r["ans"] for r in rows])
    t = json.load(open(HEADS))["idk_threshold"]

    def met(Pm, a, th):
        answer = Pm.max(1) >= th; top = Pm.argmax(1)
        corr = answer & (top == a); wrong = answer & (top != a); idk = ~answer
        acc = corr.mean() * 100; nh = 100 * (1 - wrong.sum() / max(1, wrong.sum() + idk.sum()))
        return acc, nh, (acc + nh) / 2, answer.mean() * 100

    print(f"n={len(rows)} questions | plain accuracy (always answer) {np.mean(P.argmax(1) == ans) * 100:.1f}% (chance 25%)")
    for name, th in [("always answer", 0.0), (f"Debussy rule (threshold {t}, chosen earlier)", t), ("always 'I don't know'", 1.01)]:
        a, nh, c, f = met(P, ans, th)
        print(f"  {name:42s} accuracy {a:5.1f} | non-hallucination {nh:5.1f} | component {c:5.1f} | answered {f:4.0f}%")
    print("  per domain (Debussy rule):")
    for dmn in sorted(set(r["domain"] for r in rows)):
        msk = np.array([r["domain"] == dmn for r in rows]); a, nh, c, f = met(P[msk], ans[msk], t)
        print(f"    {dmn:38s} n={msk.sum():3d} component {c:5.1f} (acc {a:4.1f}, non-halluc {nh:5.1f})")


if __name__ == "__main__":
    {"gen": lambda: gen(sys.argv[2]), "eval": lambda: evaluate(sys.argv[2], sys.argv[3]),
     "--score": lambda: score(sys.argv[2])}[sys.argv[1]]()
