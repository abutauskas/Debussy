# Releasing a new version

1. **Update the version in three places** (the release refuses to run if they disagree):
   - `debussy/__init__.py`: `__version__ = "0.1.1"`
   - `CHANGELOG.md`: a new section at the top, `## 0.1.1 (Month D, YYYY): short title`
   - `site/releases/index.html`: a new `<section id="v0.1.1">` above the previous one (move the "Latest" badge)
2. Check locally: `python scripts/release/check.py 0.1.1`, then commit and push.
3. On GitHub: **Actions → Release → Run workflow**, type `0.1.1`, **Run**.
   It creates the tag `v0.1.1`, the release "Debussy 0.1.1" with the notes from `CHANGELOG.md`, and attaches
   `debussy-0.1.1-heads.zip` (both heads files, LICENSE, NOTICE, CHANGELOG).

If you publish a release by hand instead, the same workflow attaches the zip to it. Running the workflow again for an
existing version refreshes its notes and zip (that is also how to add files to a release that has none).
