# Moving Debussy to a new base model (knowledge transfer)

Debussy's know-how lives in `engine.py` and is no longer tied to Qwen.

## What transfers unchanged
- **The decision skills** — the exact questions Debussy asks itself:
  - When2Call: the five yes/no checks (tool exists? values grounded? required values given? already provided? actually required?)
  - WildGuard: the three questions answered in one pass (harmful request → refusal → harmful reply)
  - Knowledge / HLE: one-pass letter scoring
- **The rules**: hide any "I don't know" option and abstain below a confidence threshold; never call a tool that is not in the list (opt-in)
- **The training/calibration data and procedures**: When2Call TRAIN pairs, WildGuard TRAIN calibration rows (natural-mix weighting), the MMLU-Pro knowledge proxy, train-only cross-validation for every choice
- **All evaluation sets and scorers**: When2Call test, WildGuard agreed subset, HLE (513), AA public Omniscience sample

## What is re-fitted per base model
- The small heads in `heads.json` (each model's confidence scale differs) and the "I don't know" threshold,
  using the same training data and the same train-only selection rules.

## How
1. Add a profile in `engine.py` (chat format + how to skip the model's thinking = System 1). Done for `qwen3.5` and `k2-horizon`.
2. Run the skills: `DEBUSSY_PROFILE=<profile> DEBUSSY_BACKEND=<inproc|server> python transfer_run.py <skill> ...`
3. Re-fit heads with the existing scorers (`score_w2o.py`, `score_wildguard.py --natural-weights`, `omni_proxy.py --score`).

## Verified
- `engine_check.py`: the engine reproduces Debussy's Qwen results exactly (max difference 0.0000) for When2Call, WildGuard
  and letter scoring; `transfer_run.py omni_public` reproduces the saved stand-in probabilities exactly.

## K2-Horizon-7B (IFM, Apache-2.0)
A reasoning ("System 2") model; Debussy runs it as System 1 by opening and immediately closing its thinking block
(`<ifm|think>\n</ifm|think>`). Its architecture (`k2-horizon`) is newer than the installed `llama-cpp-python`, so it runs
through a current llama.cpp `llama-server` (backend `server`).
