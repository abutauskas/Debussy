"""Score more When2Call TEST items with the same 5-check judge (bounded job; resumable).

usage: python w2c_extra.py MODEL.gguf START END OUT.json
Uses the same fixed test shuffle as w2c_options.py; START/END index into the test items.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root on the path
import json, os, sys, time
from debussy.skills.when2call import Judge, build_items

path, start, end, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
items = [it for it in build_items(150, 600) if it["split"] == "te"][start:end]
rows = json.load(open(out)) if os.path.exists(out) else []
j = Judge(path); t0 = time.time(); s0 = len(rows)
for i in range(s0, len(items)):
    it = items[i]
    avail = it["name"] in {(json.loads(x) if isinstance(x, str) else x).get("name") for x in it["tools"]}
    rows.append({"split": "te", "y": it["y"], "avail": avail, "lo": j.run(it["tools"], it["q"], it["call"], it["name"], it["rfi"])})
    if (i + 1) % 25 == 0 or i + 1 == len(items):
        json.dump(rows, open(out, "w")); print(f"{i+1}/{len(items)} {(time.time()-t0)/(i+1-s0):.2f}s/ex", flush=True)
print("DONE")
