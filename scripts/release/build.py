"""Build the files for a release: dist/debussy-<v>-heads.zip and dist/notes.md (from CHANGELOG.md).

usage: python scripts/release/build.py 0.1.1
"""
import os, re, sys, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
v = sys.argv[1]
dist = os.path.join(ROOT, "dist"); os.makedirs(dist, exist_ok=True)

log = open(os.path.join(ROOT, "CHANGELOG.md")).read()
m = re.search(rf"^## {re.escape(v)} \(.*?\n(.*?)(?=^## |\Z)", log, re.M | re.S)
notes = m.group(1).strip() + (f"\n\n**Website:** https://debussylabs.com/releases/#v{v} · "
                              "**Files:** `debussy-{v}-heads.zip` holds the calibrated heads; the base models come from Hugging Face.").replace("{v}", v)
open(os.path.join(dist, "notes.md"), "w").write(notes + "\n")

readme = f"""Debussy {v}: calibrated heads

heads.json              heads for Qwen3.5-4B   (Debussy(), and DebussyDuo tool calls)
heads_k2-horizon.json   heads for K2-Horizon-7B (safety, knowledge, letter-bias correction)

These files are also inside the source code. To use them, get the code and the base models:
  git clone https://github.com/abutauskas/Debussy && cd Debussy      (instructions in README.md)
Base model files (GGUF) come from Hugging Face: unsloth/Qwen3.5-4B-GGUF and IFM/K2-Horizon-7B-GGUF.

Apache-2.0 (LICENSE). Third-party notices: NOTICE.md. Release notes: CHANGELOG.md.
Built by Debussy Labs with help from Claude, Anthropic's AI assistant. Not an Anthropic model.
"""
name = f"debussy-{v}-heads"; out = os.path.join(dist, name + ".zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in ["debussy/heads.json", "debussy/heads_k2-horizon.json", "LICENSE", "NOTICE.md", "CHANGELOG.md"]:
        z.write(os.path.join(ROOT, rel), f"{name}/{os.path.basename(rel)}")
    z.writestr(f"{name}/README.txt", readme)
print("built", os.path.relpath(out, ROOT), "and dist/notes.md")
