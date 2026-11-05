"""Look it up, then decide: Wikipedia retrieval in front of Debussy's multiple-choice decision, honesty first.

For each Omniscience stand-in question (AA public sample, same 4 options as omni_public.py):
  1. search Wikipedia (full question, then key words if nothing comes back) and keep up to 3 short extracts;
  2. one decision pass with the notes in front of the question -> probability per option;
  3. --score applies the SAME pre-set rule as before (answer only if top probability >= 0.7, else "I don't know").
Only Wikipedia is queried; never benchmark sites or answer keys.

usage: python omni_retrieval.py run OUT.json   |   python omni_retrieval.py --score OUT.json
"""
import json, os, random, re, sys, time, urllib.parse, urllib.request
import numpy as np

from ..paths import ROOT as HERE
UA = {"User-Agent": "DebussyDecisionResearch/0.1 (offline research script; low request rate)"}
STOP = set("the a an of in on at to for from by with and or is are was were be been what which who whom whose when where how "
           "why does do did that this these those it its as into than then their there under over between during after before "
           "approximately nearest name named called considered according".split())
L = ["A", "B", "C", "D"]


def wiki(params):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {}


def keywords(q, n=8):
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", q) if w.lower() not in STOP and len(w) > 2]
    caps = [w for w in words if w[0].isupper()]
    return " ".join((caps + [w for w in words if w not in caps])[:n])


def retrieve(q, k=3, chars=600):
    titles = []
    for query in (q[:300], keywords(q, 6), keywords(q, 4), keywords(q, 3)):
        res = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": k})
        titles = [h["title"] for h in res.get("query", {}).get("search", [])]
        time.sleep(0.5)
        if titles: break
    if not titles: return []
    res = wiki({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": k, "titles": "|".join(titles)})
    time.sleep(0.5)
    pages = res.get("query", {}).get("pages", {}).values()
    by = {p.get("title"): (p.get("extract") or "").strip() for p in pages}
    return [{"title": t, "text": by.get(t, "")[:chars]} for t in titles if by.get(t)]


def run(out):
    import llama_cpp
    from llama_cpp import Llama
    items = [x for x in json.load(open(os.path.join(HERE, "data/omni_public_gen.json"))) if len(x["wrong"]) == 3]
    rows = json.load(open(out)) if os.path.exists(out) else []
    m = Llama(model_path=os.path.join(HERE, "models/Qwen3.5-4B-Q4_K_M.gguf"), n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
    ids = [m.tokenize(l.encode(), add_bos=False)[0] for l in L]; t0 = time.time(); s0 = len(rows)
    for i in range(s0, len(items)):
        x = items[i]; opts = x["wrong"] + [x["answer"]]; random.Random(x["qid"]).shuffle(opts)  # same shuffle as omni_public.py
        docs = retrieve(x["q"])
        notes = "\n\n".join(f"[{d['title']}] {d['text']}" for d in docs) or "(no reference notes found)"
        body = x["q"] + "\n\n" + "\n".join(f"{L[k]}. {o}" for k, o in enumerate(opts))
        p = ("<|im_start|>user\nReference notes from Wikipedia (they may be irrelevant or incomplete):\n" + notes +
             "\n\nAnswer the multiple-choice question with the letter of the correct option.\n\n" + body +
             "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ")
        m.reset(); m.eval(m.tokenize(p.encode(), add_bos=False, special=True)[-4000:])
        lg = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(m._ctx.ctx, -1), shape=(m.n_vocab(),)).astype(np.float64)[ids]
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        rows.append({"qid": x["qid"], "domain": x["domain"], "p": pr.round(5).tolist(), "ans": opts.index(x["answer"]),
                     "n_docs": len(docs), "titles": [d["title"] for d in docs],
                     "answer_in_notes": x["answer"].lower() in notes.lower()})
        if (i + 1) % 20 == 0 or i + 1 == len(items):
            json.dump(rows, open(out, "w")); print(f"{i+1}/{len(items)} {(time.time()-t0)/(i+1-s0):.1f}s/q", flush=True)
    print("DONE")


def score(src):
    from .omniscience import score as base_score
    rows = json.load(open(src)); base_score(src)
    found = np.mean([r["n_docs"] > 0 for r in rows]) * 100; hit = np.mean([r["answer_in_notes"] for r in rows]) * 100
    print(f"  Wikipedia returned notes for {found:.0f}% of questions; the exact answer text appeared in the notes for {hit:.0f}%")


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else score(sys.argv[2])
