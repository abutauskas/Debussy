"""Zero-shot When2Call decisions with a small open LM (CPU), scored by next-token letter probability.

usage: python lm_decide.py MODEL_ID N_SAMPLES
"""
import os
import json, random, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
_ROOT = os.path.dirname(os.path.abspath(__file__))  # project root (this folder)

torch.set_num_threads(4)
D = _ROOT + "/data/"
LABELS = ["tool_call", "request_for_info", "cannot_answer"]
LETTERS = ["A", "B", "C"]

SYSTEM = (
    "You decide how an assistant with tools should respond to a user. Choose exactly one:\n"
    "A) CALL A TOOL - one of the listed tools fits the request AND every required argument "
    "value is stated or clearly implied in the request.\n"
    "B) ASK FOR INFO - a listed tool fits the request, but a required argument value is "
    "missing from the request, so the assistant must ask the user for it.\n"
    "C) CANNOT ANSWER - none of the listed tools can do what the user wants (or there are no "
    "tools), so the assistant must say it cannot help.\n"
    "Reply with a single letter: A, B or C."
)


def compact_tool(t):
    t = json.loads(t) if isinstance(t, str) else t
    p = t.get("parameters", {}) or {}
    props = p.get("properties", {}) or {}
    req = p.get("required", t.get("required", [])) or []
    args = []
    for k, v in props.items():
        v = v or {}
        flag = "required" if k in req else "optional"
        args.append(f"    - {k} ({v.get('type','')}, {flag}): {str(v.get('description',''))[:120]}")
    return f"- {t.get('name','')}: {str(t.get('description',''))[:200]}\n" + "\n".join(args)


def build_prompt(tok, question, tools):
    tl = "\n".join(compact_tool(t) for t in tools) or "(no tools available)"
    user = f"Available tools:\n{tl}\n\nUser request: {question}\n\nWhich response is correct? Answer A, B or C."
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + "Answer: "


def load_test(n, seed=0):
    T = [json.loads(l) for l in open(D + "w2c_when2call_test_mcq.jsonl")]
    random.Random(seed).shuffle(T)
    return T[:n] if n else T


if __name__ == "__main__":
    mid, n = sys.argv[1], int(sys.argv[2])
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).eval()
    ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    ids_sp = [tok.encode(" " + l, add_special_tokens=False)[0] for l in LETTERS]
    T = load_test(n)
    correct, t0, conf = 0, time.time(), {}
    for i, t in enumerate(T):
        x = tok(build_prompt(tok, t["question"], t["tools"]), return_tensors="pt")
        with torch.inference_mode():
            lg = model(**x).logits[0, -1].float()
        sc = torch.maximum(lg[ids], lg[ids_sp])
        pred = LABELS[int(sc.argmax())]
        correct += pred == t["correct_answer"]
        conf[(t["correct_answer"], pred)] = conf.get((t["correct_answer"], pred), 0) + 1
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(T)} acc={correct/(i+1):.3f} {((time.time()-t0)/(i+1)):.1f}s/ex", flush=True)
    print(f"FINAL {mid} acc={correct/len(T):.3f} n={len(T)}")
    for k, v in sorted(conf.items()): print(k, v)
