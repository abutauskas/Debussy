"""Debussy engine: the model-agnostic core that every Debussy skill is built on.

Knowledge transfer: Debussy's skills (When2Call checks, WildGuard questions, the honesty rule, HLE/knowledge letter
scoring) only need two primitives from a base model, so moving Debussy to a new base model = adding a PROFILE (its chat
format and how to skip its thinking) and picking a BACKEND (how to run it). The small calibrated heads in heads.json
are then re-fitted on the same training data for the new base, because every model's confidence scale differs.

  Engine.yes_no(prefix, questions)       -> {name: yes-vs-no log-odds}, prefix read once, each question asked after it
  Engine.chain_yes_no(prefix, questions) -> same, but answered in one pass (Q1, then Q2 given Q1 ...) - WildGuard style
  Engine.letter_probs(user_text, letters)-> probability for each answer letter (multiple choice)

Profiles: "qwen3.5" (Qwen3.5 GGUFs, current Debussy) and "k2-horizon" (IFM K2-Horizon, a reasoning model run as System 1).
Backends: "inproc" (llama-cpp-python in this process) and "server" (a llama.cpp llama-server over HTTP; needed for
architectures newer than the installed llama-cpp-python, e.g. k2-horizon).

Select with env vars DEBUSSY_PROFILE / DEBUSSY_BACKEND / DEBUSSY_MODEL / DEBUSSY_SERVER, or pass them to get_engine().
"""
import json, os, urllib.request
import numpy as np

from .paths import ROOT as HERE

# ---- profiles: chat format + "System 1" switch (open the thinking block and close it immediately) -------------------
PROFILES = {
    "qwen3.5": {
        "model": "models/Qwen3.5-4B-Q4_K_M.gguf",
        "user_start": "<|im_start|>user\n",
        "assistant_start": "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "bos": "",
    },
    "qwen3.5-2b": {  # Debussy Mini: half the size of Qwen3.5-4B, same chat format
        "model": "models/Qwen3.5-2B-Q4_K_M.gguf",
        "user_start": "<|im_start|>user\n",
        "assistant_start": "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "bos": "",
    },
    "k2-horizon": {
        "model": "models/K2-Horizon-7B-Q4_K_M.gguf",
        "user_start": "<|ifm|im_start|>user\n",
        "assistant_start": "<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>",
        "bos": "<|ifm|begin_of_text|>",
        # K2's tokenizer writes the answer as " B" right after "Answer:" (no trailing space in the cue)
        "answer_cue": "Answer:", "letter_prefix": " ",
    },
}
YES = ("yes", "Yes", " yes", " Yes")
NO = ("no", "No", " no", " No")


class _InProc:
    """llama-cpp-python in this process (fast, supports state save/restore)."""
    def __init__(self, path, n_ctx):
        import llama_cpp
        from llama_cpp import Llama
        self._lc = llama_cpp
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=os.cpu_count() or 4, n_batch=512, verbose=False)

    def tok(self, s, special=False): return self.llm.tokenize(s.encode(), add_bos=False, special=special)
    def detok(self, ids): return self.llm.detokenize(ids).decode("utf-8", "ignore")
    def first_id(self, s): return self.tok(s)[0]

    def logits_after(self, text, keep_state=False):
        self.llm.reset(); self.llm.eval(self.tok(text, special=True)[-(self.llm.n_ctx() - 64):])
        lg = np.ctypeslib.as_array(self._lc.llama_get_logits_ith(self.llm._ctx.ctx, -1), shape=(self.llm.n_vocab(),))
        return lg.astype(np.float64)

    def score_continuations(self, prefix, suffixes, ids):
        """Read prefix once, then for each suffix return logits of `ids` after prefix+suffix."""
        self.llm.reset(); self.llm.eval(self.tok(prefix, special=True)[-(self.llm.n_ctx() - 512):])
        st = self.llm.save_state(); out = []
        for s in suffixes:
            self.llm.load_state(st); self.llm.eval(self.tok(s, special=True))
            lg = np.ctypeslib.as_array(self._lc.llama_get_logits_ith(self.llm._ctx.ctx, -1), shape=(self.llm.n_vocab(),))
            out.append(lg.astype(np.float64)[ids])
        return out

    def eval_more(self, text):  # continue the current sequence (for chained questions)
        self.llm.eval(self.tok(text, special=False))
        return np.ctypeslib.as_array(self._lc.llama_get_logits_ith(self.llm._ctx.ctx, -1), shape=(self.llm.n_vocab(),)).astype(np.float64)


class _Server:
    """A llama.cpp llama-server over HTTP. Uses top-n token log-probs of the next token; the server's prompt cache
    reuses the shared prefix between consecutive requests, so 'read once, ask many' stays cheap."""
    def __init__(self, url, n_probs=64):
        self.url = url.rstrip("/"); self.n_probs = n_probs; self._seq = ""

    def _post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    def tok(self, s, special=False):
        return self._post("/tokenize", {"content": s, "add_special": False, "parse_special": special})["tokens"]

    def first_id(self, s): return self.tok(s)[0]
    def detok(self, ids): return self._post("/detokenize", {"tokens": list(ids)})["content"]

    def _next_logprobs(self, text):
        r = self._post("/completion", {"prompt": text, "n_predict": 1, "n_probs": self.n_probs, "temperature": 0,
                                       "cache_prompt": True, "post_sampling_probs": False, "samplers": []})
        probs = r.get("completion_probabilities") or []
        top = probs[0].get("top_logprobs") or probs[0].get("probs") if probs else []
        out = {}
        for t in top or []:
            tid = t.get("id"); lp = t.get("logprob")
            if lp is None and "prob" in t: lp = float(np.log(max(t["prob"], 1e-12)))
            if tid is not None: out[int(tid)] = float(lp)
        return out

    def _vec(self, lp, ids):
        floor = (min(lp.values()) - 1.0) if lp else -30.0   # unseen tokens: below the smallest reported log-prob
        return np.array([lp.get(i, floor) for i in ids], dtype=np.float64)

    def score_continuations(self, prefix, suffixes, ids):
        return [self._vec(self._next_logprobs(prefix + s), ids) for s in suffixes]

    def logits_after(self, text, keep_state=False):
        self._seq = text; return self._next_logprobs(text)

    def eval_more(self, text):
        self._seq += text; return self._next_logprobs(self._seq)


class Engine:
    def __init__(self, profile="qwen3.5", backend="inproc", model=None, server=None, n_ctx=4096):
        self.p = PROFILES[profile]; self.profile = profile
        path = model or os.path.join(HERE, self.p["model"])
        self.b = _InProc(path, n_ctx) if backend == "inproc" else _Server(server or os.environ.get("DEBUSSY_SERVER", "http://127.0.0.1:8080"))
        self.yes = [self.b.first_id(s) for s in YES]; self.no = [self.b.first_id(s) for s in NO]

    # chat helpers (llama-server adds the model's BOS token itself, so only add it as text in-process)
    def open_user(self): return (self.p["bos"] if isinstance(self.b, _InProc) else "") + self.p["user_start"]
    def close_user(self): return self.p["assistant_start"]

    def _lo(self, v_yes, v_no): return float(np.logaddexp.reduce(v_yes) - np.logaddexp.reduce(v_no))

    def yes_no(self, prefix_user_text, questions):
        """prefix_user_text: start of the user turn (context). questions: {name: question text}."""
        ids = self.yes + self.no
        res = self.b.score_continuations(self.open_user() + prefix_user_text,
                                         [q + self.close_user() for q in questions.values()], ids)
        return {k: self._lo(v[:4], v[4:]) for k, v in zip(questions, res)}

    def chain_yes_no(self, user_text, names, first_cue="Q1:"):
        """One pass: user_text already lists the questions; the answer is read as 'Q1: yes\\nQ2: ...'."""
        ids = self.yes + self.no; out = {}
        if isinstance(self.b, _InProc):
            lg = self.b.logits_after(self.open_user() + user_text + self.close_user() + first_cue)[ids]
        else:
            lg = self.b._vec(self.b.logits_after(self.open_user() + user_text + self.close_user() + first_cue), ids)
        for i, k in enumerate(names):
            out[k] = self._lo(lg[:4], lg[4:])
            if i + 1 < len(names):
                nxt = self.b.eval_more(f"{' yes' if out[k] > 0 else ' no'}\nQ{i+2}:")
                lg = nxt[ids] if isinstance(nxt, np.ndarray) else self.b._vec(nxt, ids)
        return out

    def letter_probs(self, user_text, letters, cue=None):
        cue = self.p.get("answer_cue", "Answer: ") if cue is None else cue
        ids = [self.b.first_id(self.p.get("letter_prefix", "") + l) for l in letters]
        text = self.open_user() + user_text + self.close_user() + cue
        v = self.b.logits_after(text)
        v = v[ids] if isinstance(v, np.ndarray) else self.b._vec(v, ids)
        pr = np.exp(v - v.max()); return pr / pr.sum()


def keep_last_tokens(e, text, n):
    """Trim text to its last n tokens (by the engine's own tokenizer), marking the cut."""
    t = e.b.tok(text)
    return text if len(t) <= n else "[...]\n" + e.b.detok(t[-n:])


def get_engine(**kw):
    return Engine(profile=kw.get("profile", os.environ.get("DEBUSSY_PROFILE", "qwen3.5")),
                  backend=kw.get("backend", os.environ.get("DEBUSSY_BACKEND", "inproc")),
                  model=kw.get("model", os.environ.get("DEBUSSY_MODEL")), server=kw.get("server"),
                  n_ctx=kw.get("n_ctx", int(os.environ.get("DEBUSSY_NCTX", "4096"))))   # in-process context size
