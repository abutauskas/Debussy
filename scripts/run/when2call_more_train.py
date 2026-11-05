"""Add more When2Call TRAIN pairs (tool_call vs request_for_info) not used before. Bounded, resumable.

usage: python w2c_more_train.py MODEL.gguf NEW_PER_CLASS OUT.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
import json, os, sys, time
from debussy.skills.when2call import Judge, build_items

path, extra, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
base = int(os.environ.get("W2C_BASE", "150"))  # pairs per class already used (150 initial, 250 after batch 1)
old = {(it["q"], it["call"]) for it in build_items(base, 0)}
items = [it for it in build_items(base + extra, 0) if (it["q"], it["call"]) not in old]
rows = json.load(open(out)) if os.path.exists(out) else []
j = Judge(path); t0 = time.time(); s0 = len(rows)
for i in range(s0, len(items)):
    it = items[i]
    avail = it["name"] in {(json.loads(x) if isinstance(x, str) else x).get("name") for x in it["tools"]}
    rows.append({"split": "tr", "y": it["y"], "avail": avail, "lo": j.run(it["tools"], it["q"], it["call"], it["name"], it["rfi"])})
    if (i + 1) % 25 == 0 or i + 1 == len(items):
        json.dump(rows, open(out, "w")); print(f"{i+1}/{len(items)} {(time.time()-t0)/(i+1-s0):.2f}s/ex", flush=True)
print("DONE")
