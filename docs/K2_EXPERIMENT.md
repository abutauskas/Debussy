# Debussy on K2-Horizon-7B: a System 2 model run as System 1

**Question:** does swapping Debussy's base model from Qwen3.5-4B to K2-Horizon-7B (IFM, Apache-2.0, a reasoning
model nobody else uses for decision models) make Debussy better, when K2 is run with its thinking switched off?

**How:** K2 runs through llama.cpp's `llama-server` with its thinking block opened and closed immediately
(`<ifm|think>\n</ifm|think>`), so every decision is a single pass. Debussy's skills are unchanged (see `TRANSFER.md`);
only the small calibrated heads are re-fitted on K2, using the same training data and train-only selection rules.
Every comparison below uses **identical training/calibration rows and identical test rows** for both models.

## Results

| Benchmark (25% each) | Qwen3.5-4B | K2-Horizon-7B | Test rows | Note |
|---|---|---|---|---|
| When2Call (model-only) | **94.0%** | 89.3% | 300 | Qwen with 799 training pairs, K2 with 500 (K2 did not improve from 200 → 500) |
| WildGuardTest (avg F1) | 89.8 | **92.8** | 300 | both calibrated on the same 400 rows (150 balanced + 250 natural mix) |
| HLE-MultiChoice | 10.5% | **13.6%** | all 513 | chance 16.4%; both below chance, K2 less so |
| Omniscience stand-in | 51.8 | **53.1** | 266 | each with its own "I don't know" threshold from the practice set (Qwen 0.7, K2 0.9) |

**Index estimates** (0.25 × the four parts; the Omniscience part is a public stand-in):

| Debussy variant | When2Call | WildGuard | HLE | Omniscience | **Index** |
|---|---|---|---|---|---|
| Qwen3.5-4B (current default) | 93.2 (600 q) | 89.9 (880 rows) | 10.5 | 51.8 | **≈ 61.4** |
| K2-Horizon-7B | 89.3 | 92.8 | 13.6 | 53.1 | **≈ 62.2** |
| Two specialists: Qwen for tool calls, K2 for safety + knowledge | 93.2 | 92.8 | 13.6 | 53.1 | **≈ 63.2** |

## What we learned
- **K2 is the better judge and the more honest knower; Qwen is the better tool-caller.** K2's raw safety judgements are
  clearly sharper (harmful reply 0.87 vs 0.80, refusal 0.86 vs 0.81, before calibration, 300 test rows); Qwen's tool-call instincts are better, and more
  training pairs did not close K2's gap.
- **Calibration matters as much as the model.** With 150 calibration rows K2 merely tied Qwen on WildGuard (0.873 vs 0.871);
  with the same natural-mix rows Qwen got, K2 pulled ahead (0.928 vs 0.898).
- **K2 is overconfident.** Its own "I don't know" threshold is 0.9 (Qwen: 0.7). With it, K2 answers only 9% of obscure
  questions and is honest on 98% of the rest.
- **Small-sample swings are real.** K2's HLE lead looked like +5 on the first 150 questions and settled at +3 over 513.
- Fixes found along the way (they help both models): a train-chosen "tool exists" cut-off (Qwen 92.8 → 93.2% on When2Call),
  token-based trimming of over-long inputs, and llama-server settings that fit 15 GB of RAM (`--cache-ram 0`).

## Costs and caveats
- K2 is a 7B model: about 1.75× slower than Qwen3.5-4B on this CPU (a When2Call decision ~10 s vs ~6 s), and it needs a
  current llama.cpp build (`llama-server`), because the installed `llama-cpp-python` cannot load its architecture.
- The "two specialists" variant loads two models (about 9 GB of RAM) but still makes each decision in a single pass.
- Sample sizes: 300 test rows for When2Call/WildGuard (±~2-3 points), all 513 for HLE (±~3); the Omniscience part is a
  stand-in built from Artificial Analysis's public sample with model-written wrong options.
- These are our estimates, not official Artificial Analysis scores.

## Reproduce (K2 needs a running llama-server; see `debussy/engine.py`)
```bash
export DEBUSSY_PROFILE=k2-horizon DEBUSSY_BACKEND=server
python scripts/run/transfer_run.py hle                          # all 513
python scripts/run/transfer_run.py omni_public && python scripts/run/transfer_run.py mmlu_proxy 400
python scripts/run/transfer_run.py wg 150 300 && python scripts/run/transfer_run.py wg_natural 170 1 && python scripts/run/transfer_run.py wg_natural 80 2
python scripts/run/transfer_run.py w2c 100 150 && python scripts/run/transfer_run.py w2c_more_train 100 250 && python scripts/run/transfer_run.py w2c_more_test 150 300
python scripts/score/compare_bases.py wg                        # identical rows for both models
```
