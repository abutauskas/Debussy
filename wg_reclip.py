"""Re-run only the WildGuard rows that were clipped at 600 tokens, with a longer clip (WG_CLIP). Resumable.

usage: WG_CLIP=1500 python wg_reclip.py IN.json OUT.json
Rows that were never clipped are copied unchanged (their prompt text is identical at any clip length).
"""
import json, os, sys, time
from wg_rows import all_rows
from wildguard_yesno import Judge
_ROOT = os.path.dirname(os.path.abspath(__file__))  # project root (this folder)

src, out = sys.argv[1], sys.argv[2]
rows = json.load(open(src)); R = all_rows(); assert len(rows) == len(R)
j = Judge(_ROOT + "/models/Qwen3.5-4B-Q4_K_M.gguf")
L = lambda s: len(j.llm.tokenize(str(s).encode(), add_bos=False, special=False))
todo = [i for i, (_, r) in enumerate(R) if L(r["prompt"]) > 600 or L(r["response"]) > 600]
done = json.load(open(out + ".partial")) if os.path.exists(out + ".partial") else {}
t0 = time.time(); n0 = len(done)
for n, i in enumerate(todo):
    if str(i) in done: continue
    done[str(i)] = j.probs(R[i][1]["prompt"], R[i][1]["response"])
    if len(done) % 10 == 0 or len(done) == len(todo):
        json.dump(done, open(out + ".partial", "w")); print(f"{len(done)}/{len(todo)} {(time.time()-t0)/(len(done)-n0):.1f}s/row", flush=True)
for i, lo in done.items(): rows[int(i)] = dict(rows[int(i)], lo=lo, reclipped=True)
json.dump(rows, open(out, "w")); print("DONE", len(done), "rows re-run")
