# Contributing to Debussy

Thanks for helping. Debussy is maintained by Debussy Labs; changes come in as **pull requests from a fork**
(only the maintainers push to the main branch).

## How to contribute
1. Fork the repository and create a branch.
2. Make your change, with a short explanation of *why* in the pull request.
3. Run the checks that apply (below), and say in the pull request which ones you ran.
4. Open a pull request. A maintainer reviews and merges it.

For a big change (a new base model, a new skill, a change to how results are measured), open an issue first so we can
agree on the approach.

## The rules that keep Debussy's numbers honest
- **Never choose anything on test data.** Heads, thresholds, prompts and settings are chosen on training or practice
  data only; a test set is used once, to report. If you change the rule for adopting something, fix it *before* you look
  at test results.
- **Report results as measured**, including the ones that didn't work. Negative results go in `docs/` and the changelog.
- **Keep it System 1**: one pass per decision, no step-by-step reasoning or long generations.
- **No gated data, no per-question results, no secrets** in the repository. WildGuardMix and Humanity's Last Exam are
  gated; HLE questions must never be published or trained on. API keys and tokens never go in code or commits.
- Debussy is credited as "built with help from Claude, not an Anthropic model". Please keep that wording accurate.

## Checks
- Package = research pipeline: `python scripts/checks/parity_check.py 5` (and `... 5 k2-horizon server` for K2).
- API server: `python -m debussy.server --fake` and call the endpoints in `site/docs/`.
- Website: after editing `scripts/site/header.html` or `footer.html`, run `python scripts/site/stamp.py`.
- Sign-in, keys and gateway: `cloudflare/test/README.md`.
- Releases: `docs/RELEASING.md` (maintainers).

## Licence
By contributing you agree that your contribution is licensed under the project's licence, Apache-2.0 (`LICENSE`).
