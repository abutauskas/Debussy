"""LoRA fine-tune a small LM to make When2Call decisions on CPU, then score on the test split.

Training data: When2Call TRAIN only (pref file), plus "near-miss" augmentation built from train
tool-call examples: drop the tool that was called, keep the other tools -> label cannot_answer.
The test split is only used for scoring.

usage: python train_lora.py MODEL_ID N_PER_CLASS N_NEARMISS N_TEST OUT_DIR [MAX_STEPS]
"""
import json, random, re, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from lm_decide import build_prompt, LABELS, LETTERS, D
from when2call_head import label_pref

torch.set_num_threads(4)
torch.manual_seed(0)
mid, nper, nnear, nte, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
max_steps = int(sys.argv[6]) if len(sys.argv) > 6 else None

# ---- training data (train split only) ----
P = [json.loads(l) for l in open(D + "w2c_when2call_train_pref.jsonl")]
random.Random(2).shuffle(P)
data, per, near = [], {0: 0, 1: 0, 2: 0}, 0
for p in P:
    c = p["chosen_response"]["content"]; y = label_pref(c); q = p["messages"][-1]["content"]
    if y is None: continue
    if per[y] < nper:
        per[y] += 1; data.append((q, p["tools"], y))
    elif y == 0 and near < nnear and len(p["tools"]) > 1:
        called = set(re.findall(r'"name":\s*"([^"]+)"', c.split("<TOOLCALL>")[-1]))
        rest = [t for t in p["tools"] if json.loads(t).get("name") not in called]
        if rest and len(rest) < len(p["tools"]):
            near += 1; data.append((q, rest, 2))
random.Random(3).shuffle(data)
print(f"train examples: {len(data)} per-class={per} near-miss={near}", flush=True)

tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()
letter_ids = torch.tensor([tok.encode(l, add_special_tokens=False)[0] for l in LETTERS])

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
ACC = 8
steps_total = (len(data) + ACC - 1) // ACC if max_steps is None else max_steps
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / steps_total))
model.train(); t0 = time.time(); run_loss = 0.0; step = 0
for i, (q, tools, y) in enumerate(data):
    x = tok(build_prompt(tok, q, tools), return_tensors="pt", truncation=True, max_length=768)
    logits = model(**x, logits_to_keep=1).logits[0, -1]
    # restrict to the three answer letters: a clean 3-way classification loss
    loss = torch.nn.functional.cross_entropy(logits[letter_ids][None], torch.tensor([y])) / ACC
    loss.backward(); run_loss += loss.item()
    if (i + 1) % ACC == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(); step += 1
        if step % 5 == 0:
            print(f"step {step}/{steps_total} loss={run_loss/5:.3f} {(time.time()-t0)/(i+1):.2f}s/ex", flush=True); run_loss = 0.0
        if max_steps and step >= max_steps: break
model.save_pretrained(out)
print(f"trained in {(time.time()-t0)/60:.1f} min", flush=True)

# ---- score on When2Call test (same fixed shuffle as the other experiments) ----
model.eval()
T = [json.loads(l) for l in open(D + "w2c_when2call_test_mcq.jsonl")]
random.Random(0).shuffle(T); T = T[:nte]
correct, conf = 0, {}
for i, t in enumerate(T):
    x = tok(build_prompt(tok, t["question"], t["tools"]), return_tensors="pt", truncation=True, max_length=2048)
    with torch.inference_mode():
        pred = LABELS[int(model(**x, logits_to_keep=1).logits[0, -1][letter_ids].argmax())]
    correct += pred == t["correct_answer"]; conf[(t["correct_answer"], pred)] = conf.get((t["correct_answer"], pred), 0) + 1
    if (i + 1) % 100 == 0: print(f"test {i+1}/{len(T)} acc={correct/(i+1):.3f}", flush=True)
print(f"FINAL test acc={correct/len(T):.3f} n={len(T)}")
for k, v in sorted(conf.items()): print(k, v)
