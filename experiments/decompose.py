"""When2Call via decomposed yes/no judgements (System 1: several tiny single-pass checks).

For each request:
  fit(tool)      = P(yes | "Can this tool do what the user asks?")          for every tool
  given(param)   = P(yes | "Did the user give a value for <param>?")       for the best tool's required params
A tiny classifier on these ~12 numbers picks tool_call / request_for_info / cannot_answer.

usage: python decompose.py BACKEND MODEL N_PER_CLASS N_NEARMISS N_TEST OUT.npz
  BACKEND: hf (transformers) or gguf (llama.cpp; MODEL is a .gguf path)
"""
import json, os, random, re, sys, time
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

D = _ROOT + "/data/"
LABELS = ["tool_call", "request_for_info", "cannot_answer"]
MAX_PARAMS = 6


def parse_tool(t):
    t = json.loads(t) if isinstance(t, str) else t
    p = t.get("parameters", {}) or {}
    props = p.get("properties", {}) or {}
    req = p.get("required", t.get("required", [])) or []
    return t.get("name", ""), str(t.get("description", ""))[:300], props, [r for r in req if r in props or True]


def label_pref(t):
    if "<TOOLCALL>" in t: return 0
    s = t.strip()
    if re.search(r"\b(unable|can't|cannot|can not|don't have|do not have|not able|isn't possible|beyond my)\b", s, re.I) and not s.endswith("?"):
        return 2
    if "?" in s: return 1
    return None


FIT_Q = ("Tool: {name}\nWhat it does: {desc}\n\nUser request: {q}\n\n"
         "Is this tool the right tool to carry out what the user is asking for "
         "(ignoring whether all details are given)? Answer yes or no.")
PARAM_Q = ("User request: {q}\n\nTo use the tool \"{name}\" the assistant must fill in the parameter "
           "\"{p}\" ({ptype}): {pdesc}\n\nDoes the user's request state, or make completely obvious, a "
           "specific value for \"{p}\"? Answer yes or no.")


class HFBackend:
    def __init__(self, mid):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_num_threads(4)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).eval()
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)[0]
        self.yes = [enc(s) for s in ("yes", "Yes", " yes", " Yes")]
        self.no = [enc(s) for s in ("no", "No", " no", " No")]

    def prompt(self, text):
        m = [{"role": "user", "content": text}]
        try:
            return self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    def p_yes(self, text):
        x = self.tok(self.prompt(text), return_tensors="pt", truncation=True, max_length=1024)
        with self.torch.inference_mode():
            lg = self.model(**x, logits_to_keep=1).logits[0, -1].float()
        y = self.torch.logsumexp(lg[self.yes], 0); n = self.torch.logsumexp(lg[self.no], 0)
        return float(self.torch.sigmoid(y - n))


class GGUFBackend:
    def __init__(self, path):
        import llama_cpp
        from llama_cpp import Llama
        self._lc = llama_cpp
        self.llm = Llama(model_path=path, n_ctx=2048, n_threads=4, n_batch=512, logits_all=False, verbose=False)
        tk = lambda s: self.llm.tokenize(s.encode(), add_bos=False, special=False)[0]
        self.yes = [tk(s) for s in ("yes", "Yes", " yes", " Yes")]
        self.no = [tk(s) for s in ("no", "No", " no", " No")]

    def p_yes(self, text):
        # Qwen chat template with thinking disabled (empty think block)
        p = (f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        toks = self.llm.tokenize(p.encode(), add_bos=False, special=True)[-2000:]
        self.llm.reset(); self.llm.eval(toks)
        ptr = self._lc.llama_get_logits_ith(self.llm._ctx.ctx, -1)
        lg = np.ctypeslib.as_array(ptr, shape=(self.llm.n_vocab(),)).astype(np.float64)
        y = np.logaddexp.reduce(lg[self.yes]); n = np.logaddexp.reduce(lg[self.no])
        return float(1 / (1 + np.exp(n - y)))


def featurize(be, q, tools):
    parsed = [parse_tool(t) for t in tools]
    if not parsed:
        return [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], {}
    fits = [be.p_yes(FIT_Q.format(name=n, desc=d, q=q)) for n, d, _, _ in parsed]
    order = np.argsort(fits)[::-1]
    n, d, props, req = parsed[order[0]]
    req = [r for r in req][:MAX_PARAMS]
    given = []
    for r in req:
        pr = props.get(r) or {}
        given.append(be.p_yes(PARAM_Q.format(q=q, name=n, p=r, ptype=pr.get("type", ""),
                                             pdesc=str(pr.get("description", ""))[:200])))
    g = np.array(given) if given else np.array([1.0])
    f1 = fits[order[0]]; f2 = fits[order[1]] if len(fits) > 1 else 0.0
    feats = [1, len(fits) / 10, f1, f2, float(np.mean(fits)), len(req) / 6,
             float(g.min()), float(g.mean()), float(np.prod(g)), float((g < 0.5).sum()) / 6, f1 * float(g.min())]
    return feats, {"fits": fits, "given": given}


if __name__ == "__main__":
    backend, mid, nper, nnear, nte, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    be = HFBackend(mid) if backend == "hf" else GGUFBackend(mid)

    P = [json.loads(l) for l in open(D + "w2c_when2call_train_pref.jsonl")]
    random.Random(1).shuffle(P)
    train, per, near = [], {0: 0, 1: 0, 2: 0}, 0
    for p in P:
        c = p["chosen_response"]["content"]; y = label_pref(c); q = p["messages"][-1]["content"]
        if y is None: continue
        if per[y] < nper:
            per[y] += 1; train.append((q, p["tools"], y))
        elif y == 0 and near < nnear and len(p["tools"]) > 1:
            called = set(re.findall(r'"name":\s*"([^"]+)"', c.split("<TOOLCALL>")[-1]))
            rest = [t for t in p["tools"] if json.loads(t).get("name") not in called]
            if rest and len(rest) < len(p["tools"]):
                near += 1; train.append((q, rest, 2))
    T = [json.loads(l) for l in open(D + "w2c_when2call_test_mcq.jsonl")]
    random.Random(0).shuffle(T)
    test = [(t["question"], t["tools"], LABELS.index(t["correct_answer"])) for t in T[:nte]]
    items = [("tr", *x) for x in train] + [("te", *x) for x in test]

    part = out + ".partial.json"
    rows = json.load(open(part)) if os.path.exists(part) else []
    t0 = time.time(); start = len(rows)
    for i in range(start, len(items)):
        s, q, tools, y = items[i]
        f, raw = featurize(be, q, tools)
        rows.append({"split": s, "y": y, "f": f, "raw": raw})
        if (i + 1) % 50 == 0 or i + 1 == len(items):
            json.dump(rows, open(part, "w"))
            print(f"{i+1}/{len(items)} {(time.time()-t0)/(i+1-start):.2f}s/ex", flush=True)
    X = np.array([r["f"] for r in rows]); y = np.array([r["y"] for r in rows]); sp = np.array([r["split"] for r in rows])
    np.savez_compressed(out, Xtr=X[sp == "tr"], ytr=y[sp == "tr"], Xte=X[sp == "te"], yte=y[sp == "te"])
    print("DONE", X.shape)
