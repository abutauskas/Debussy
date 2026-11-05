"""When2Call, AA-style: the model sees the tools, the request and the candidate replies.

One read of (tools + request), then three short yes/no checks from the saved state:
  exist: is the proposed tool-call's function among the available tools?
  args : is every argument value in the proposed call stated/implied by the user (not invented)?
  rfi  : has the user already given the information the "ask for info" reply asks for?
Decision: tool unavailable -> cannot_answer; else a 2-feature logistic regression (fit on TRAIN pairs)
chooses tool_call vs request_for_info.

usage: python w2c_options.py MODEL.gguf N_TRAIN_PER_CLASS N_TEST OUT.json
"""
import json, os, random, re, sys, time
import numpy as np
import llama_cpp
from llama_cpp import Llama
_ROOT = os.path.dirname(os.path.abspath(__file__))  # project root (this folder)

D = _ROOT + "/data/"
LABELS = ["tool_call", "request_for_info", "cannot_answer"]


def lab(t):
    if "<TOOLCALL>" in t: return "tc"
    s = t.strip()
    if re.search(r"\b(unable|can't|cannot|can not|don't have|do not have|not able|isn't possible|beyond my)\b", s, re.I) and not s.endswith("?"):
        return "ca"
    return "rfi" if "?" in s else None


def norm_call(t):
    """-> (json string of the first call, function name)"""
    s = t.split("<TOOLCALL>")[-1].split("</TOOLCALL>")[0] if "<TOOLCALL>" in t else t
    try:
        c = json.loads(s)
        c = c[0] if isinstance(c, list) else c
        return json.dumps(c, ensure_ascii=False), c.get("name", "")
    except Exception:
        m = re.search(r'"name":\s*"([^"]+)"', s)
        return s, (m.group(1) if m else "")


def tool_lines(tools):
    out = []
    for t in tools:
        t = json.loads(t) if isinstance(t, str) else t
        p = t.get("parameters", {}) or {}
        req = p.get("required", t.get("required", [])) or []
        out.append(f"- {t.get('name','')}: {str(t.get('description',''))[:160]} (required: {', '.join(req) or 'none'})")
    return "\n".join(out) or "(no tools available)"


Q = {
    "exist": "Proposed tool call: {call}\n\nIs the function \"{name}\" one of the available tools listed above? Answer yes or no.",
    "args": "Proposed tool call: {call}\n\nIs every argument value in this call explicitly stated in, or directly implied by, the "
            "user's request, rather than guessed or made up? Answer yes or no.",
    "rfi": "The assistant is considering replying: \"{rfi}\"\n\nHas the user's request already provided the information "
           "the assistant asks for here? Answer yes or no.",
    "req_args": "The tool \"{name}\" has these required parameters: {req}.\n\nDoes the user's request give, or make "
                "obvious, a value for every one of these required parameters? Answer yes or no.",
    "rfi_needed": "The assistant is considering replying: \"{rfi}\"\n\nIs the information it asks for strictly necessary "
                  "to carry out the request, i.e. a required parameter of the tool that the user has not given, rather "
                  "than something optional or already covered? Answer yes or no.",
}


class Judge:
    def __init__(self, path):
        self.llm = Llama(model_path=path, n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
        tk = lambda s: self.llm.tokenize(s.encode(), add_bos=False, special=False)[0]
        self.yes = [tk(s) for s in ("yes", "Yes", " yes", " Yes")]
        self.no = [tk(s) for s in ("no", "No", " no", " No")]

    def run(self, tools, q, call, name, rfi):
        req = "unknown (tool not in the list)"
        for t in tools:
            t = json.loads(t) if isinstance(t, str) else t
            if t.get("name") == name:
                p = t.get("parameters", {}) or {}
                req = ", ".join(p.get("required", t.get("required", [])) or []) or "none"
        pre = f"<|im_start|>user\nAvailable tools:\n{tool_lines(tools)}\n\nUser request: {q}\n\n"
        toks = self.llm.tokenize(pre.encode(), add_bos=False, special=True)[-3000:]
        self.llm.reset(); self.llm.eval(toks); st = self.llm.save_state(); out = {}
        for k, tmpl in Q.items():
            self.llm.load_state(st)
            suf = tmpl.format(call=call[:600], name=name, rfi=rfi[:400], req=req) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.llm.eval(self.llm.tokenize(suf.encode(), add_bos=False, special=True))
            lg = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(self.llm._ctx.ctx, -1), shape=(self.llm.n_vocab(),)).astype(np.float64)
            out[k] = float(np.logaddexp.reduce(lg[self.yes]) - np.logaddexp.reduce(lg[self.no]))
        return out


def build_items(ntr, nte):
    P = [json.loads(l) for l in open(D + "w2c_when2call_train_pref.jsonl")]
    random.Random(4).shuffle(P)
    tr, per = [], {"tc": 0, "rfi": 0}
    for p in P:
        a, b = lab(p["chosen_response"]["content"]), lab(p["rejected_response"]["content"])
        if {a, b} == {"tc", "rfi"} and per[a] < ntr:
            per[a] += 1
            tc = p["chosen_response"]["content"] if a == "tc" else p["rejected_response"]["content"]
            rf = p["chosen_response"]["content"] if a == "rfi" else p["rejected_response"]["content"]
            call, name = norm_call(tc)
            tr.append({"split": "tr", "q": p["messages"][-1]["content"], "tools": p["tools"], "call": call, "name": name,
                       "rfi": rf, "y": LABELS.index("tool_call" if a == "tc" else "request_for_info")})
    T = [json.loads(l) for l in open(D + "w2c_when2call_test_mcq.jsonl")]
    random.Random(0).shuffle(T)
    te = []
    for t in T[:nte]:
        call, name = norm_call(t["answers"]["tool_call"])
        te.append({"split": "te", "q": t["question"], "tools": t["tools"], "call": call, "name": name,
                   "rfi": t["answers"]["request_for_info"], "y": LABELS.index(t["correct_answer"])})
    return tr + te


if __name__ == "__main__":
    path, ntr, nte, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    items = build_items(ntr, nte)
    rows = json.load(open(out)) if os.path.exists(out) else []
    j = Judge(path); t0 = time.time(); start = len(rows)
    for i in range(start, len(items)):
        it = items[i]
        avail = it["name"] in {(json.loads(x) if isinstance(x, str) else x).get("name") for x in it["tools"]}
        rows.append({"split": it["split"], "y": it["y"], "avail": avail,
                     "lo": j.run(it["tools"], it["q"], it["call"], it["name"], it["rfi"])})
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            json.dump(rows, open(out, "w")); print(f"{i+1}/{len(items)} {(time.time()-t0)/(i+1-start):.2f}s/ex", flush=True)
    print("DONE")
