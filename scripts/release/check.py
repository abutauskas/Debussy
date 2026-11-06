"""Before a release: the version must match everywhere, so nothing ships half-updated.

usage: python scripts/release/check.py 0.1.1
Checks: debussy/__init__.py __version__, a "## 0.1.1 (...)" section in CHANGELOG.md, and the Releases page
(site/releases/index.html has id="v0.1.1"). Exits 1 with a list of what to fix.
"""
import os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
v = sys.argv[1] if len(sys.argv) > 1 else ""
problems = []
if not re.fullmatch(r"\d+\.\d+\.\d+", v):
    problems.append(f"'{v}' is not a version like 0.1.1")
else:
    init = open(os.path.join(ROOT, "debussy", "__init__.py")).read()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    if not m or m.group(1) != v:
        problems.append(f"debussy/__init__.py has __version__ = {m.group(1) if m else None!r}, not {v!r}")
    if not re.search(rf"^## {re.escape(v)} \(", open(os.path.join(ROOT, "CHANGELOG.md")).read(), re.M):
        problems.append(f"CHANGELOG.md has no '## {v} (date): ...' section")
    if f'id="v{v}"' not in open(os.path.join(ROOT, "site", "releases", "index.html")).read():
        problems.append(f"site/releases/index.html has no section with id=\"v{v}\"")
if problems:
    print("Not ready to release:\n" + "\n".join(f"  - {p}" for p in problems)); sys.exit(1)
print(f"Ready to release {v}: version, CHANGELOG and Releases page agree.")
