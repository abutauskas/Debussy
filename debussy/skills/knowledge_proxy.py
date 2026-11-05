"""Proxy for AA-Omniscience-Decision: 4-answer knowledge MCQ + "I don't know", scored AA-style.

AA metric: accuracy (share correct) and non-hallucination = 100*(1 - wrong/(wrong + idk)), 50/50 weighted.
We build 4-option questions from MMLU-Pro knowledge categories (correct answer + 3 random distractors),
get the model's probability for each answer in one pass, and answer only when the top probability
clears a threshold; otherwise "I don't know". The threshold is chosen on a SEPARATE calibration split.

usage: python omni_proxy.py MODEL.gguf N_QUESTIONS OUT.json      (writes per-question probabilities)
       python omni_proxy.py --score OUT.json                     (threshold analysis)
"""
import json, os, random, sys, time
import numpy as np
from ..paths import ROOT as _ROOT

D = _ROOT + "/data/"
CATS = ["history", "law", "health", "biology", "psychology", "philosophy", "other", "economics", "business"]
L = ["A", "B", "C", "D"]


def build(n):
    import pandas as pd
    d = pd.read_parquet(D + "mmlupro_test.parquet")
    d = d[d.category.isin(CATS) & (d.options.apply(len) >= 4)].sample(n, random_state=0)
    rnd = random.Random(0); qs = []
    for _, r in d.iterrows():
        opts = list(r["options"]); corr = opts[r["answer_index"]]
        wrong = [o for i, o in enumerate(opts) if i != r["answer_index"]]
        four = rnd.sample(wrong, 3) + [corr]; rnd.shuffle(four)
        qs.append({"q": r["question"], "opts": four, "ans": four.index(corr), "cat": r["category"]})
    return qs


def metrics(P, ans, t):
    top = P.argmax(1); conf = P.max(1); answer = conf >= t
    correct = answer & (top == ans); wrong = answer & (top != ans); idk = ~answer
    acc = correct.mean() * 100
    nh = 100 * (1 - wrong.sum() / max(1, wrong.sum() + idk.sum()))
    return acc, nh, (acc + nh) / 2, answer.mean() * 100


if __name__ == "__main__":
    if sys.argv[1] == "--score":
        rows = json.load(open(sys.argv[2]))
        P = np.array([r["p"] for r in rows]); ans = np.array([r["ans"] for r in rows])
        PFULL = P
        if P.shape[1] == 5: P = P[:, :4] / P[:, :4].sum(1, keepdims=True)  # A-D renormalised for the threshold table
        half = len(rows) // 2  # calibrate on first half, report on second half
        ths = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
        print(f"n={len(rows)}  plain accuracy={np.mean(P.argmax(1)==ans)*100:.1f}%")
        print(" thr   acc  non-halluc  component  answered%   (calibration half)")
        for t in ths: print(f"{t:4.2f} " + " ".join(f"{v:7.1f}" for v in metrics(P[:half], ans[:half], t)))
        if PFULL.shape[1] == 5:  # model saw an explicit "E. I don't know" option
            pE = PFULL[:, 4]; P4 = P
            def m5(idx, t, mode):
                # mode 'argmax': pick E iff it is the top option; 'pE': abstain iff p(E) > t; 'max4': threshold on renormalised A-D
                top = P4[idx].argmax(1)
                if mode == "argmax": answer = PFULL[idx].argmax(1) != 4
                elif mode == "pE": answer = pE[idx] <= t
                else: answer = P4[idx].max(1) >= t
                corr = answer & (top == ans[idx]); wrong = answer & (top != ans[idx]); idk = ~answer
                acc = corr.mean() * 100; nh = 100 * (1 - wrong.sum() / max(1, wrong.sum() + idk.sum()))
                return acc, nh, (acc + nh) / 2, answer.mean() * 100
            cal, hold = np.arange(half), np.arange(half, len(rows))
            print("\nwith explicit E option:  mode  thr  -> held-out acc / non-halluc / component / answered%")
            cands = [("argmax", 0)] + [("pE", t) for t in (0.05, 0.1, 0.2, 0.3, 0.5)] + [("max4", t) for t in ths]
            for mode, t in cands[:6]:
                print(f"  {mode:6s} {t:4.2f} cal-component {m5(cal, t, mode)[2]:5.1f}")
            bm, bt = max(cands, key=lambda c: m5(cal, c[1], c[0])[2])
            a, nh, c, f = m5(hold, bt, bm)
            print(f"chosen on calibration half: {bm} thr={bt} -> held-out acc={a:.1f} non-halluc={nh:.1f} component={c:.1f} answered={f:.0f}%")
        best = max(ths, key=lambda t: metrics(P[:half], ans[:half], t)[2])
        a, nh, c, f = metrics(P[half:], ans[half:], best)
        print(f"plain threshold policy: chosen thr={best} on held-out half: acc={a:.1f} non-halluc={nh:.1f} component={c:.1f} answered={f:.0f}%")
        sys.exit()
    import llama_cpp
    from llama_cpp import Llama
    path, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    qs = build(n)
    llm = Llama(model_path=path, n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
    IDK = os.environ.get("OMNI_IDK") == "1"
    LL = L + ["E"] if IDK else L
    ids = [llm.tokenize(l.encode(), add_bos=False)[0] for l in LL]
    rows = json.load(open(out)) if os.path.exists(out) else []
    t0 = time.time(); start = len(rows)
    for i in range(start, len(qs)):
        q = qs[i]
        opts = q["opts"] + (["I don't know"] if IDK else [])
        body = q["q"] + "\n\n" + "\n".join(f"{LL[k]}. {o}" for k, o in enumerate(opts))
        instr = ("Answer the multiple-choice question with the letter of the correct option. If you are not sure of "
                 "the answer, choose E (I don't know): a wrong answer is worse than admitting you don't know."
                 if IDK else "Answer the multiple-choice question with the letter of the correct option.")
        p = (f"<|im_start|>user\n{instr}\n\n{body}"
             "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ")
        llm.reset(); llm.eval(llm.tokenize(p.encode(), add_bos=False, special=True))
        lg = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1), shape=(llm.n_vocab(),)).astype(np.float64)[ids]
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        rows.append({"p": pr.tolist(), "ans": q["ans"], "cat": q["cat"]})
        if (i + 1) % 50 == 0 or i + 1 == len(qs):
            json.dump(rows, open(out, "w")); print(f"{i+1}/{len(qs)} {(time.time()-t0)/(i+1-start):.2f}s/ex", flush=True)
    print("DONE")
