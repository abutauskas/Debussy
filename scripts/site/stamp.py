"""Keep the header and footer identical on every page of the static site (site/**/*.html).

The site has no build step (Cloudflare Pages serves site/ as is), so after editing scripts/site/header.html or
footer.html, run:  python scripts/site/stamp.py   and commit the updated pages.
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
PARTS = {"header": (HERE / "header.html").read_text().strip(), "footer": (HERE / "footer.html").read_text().strip()}
PATTERNS = {  # an existing marker block, or (first time) the bare element
    "header": re.compile(r'<!-- header -->.*?<!-- /header -->|<header class="nav">.*?</header>', re.S),
    "footer": re.compile(r'<!-- footer -->.*?<!-- /footer -->|<footer>.*?</footer>', re.S),
}
changed = 0
for page in sorted((ROOT / "site").rglob("*.html")):
    s = t = page.read_text()
    for name, pat in PATTERNS.items():
        block = f"<!-- {name} -->\n{PARTS[name]}\n<!-- /{name} -->"
        assert len(pat.findall(t)) == 1, f"{page}: expected one {name}"
        t = pat.sub(lambda m: block, t)
    if t != s:
        page.write_text(t); changed += 1
    print(("updated  " if t != s else "unchanged") + " " + str(page.relative_to(ROOT)))
print(f"{changed} page(s) updated")
