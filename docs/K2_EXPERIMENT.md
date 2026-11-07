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
| WildGuardTest (avg F1) | 90.5 | **93.6** | 500 | both calibrated on the same 718 rows (468 balanced + 250 natural mix) |
| HLE-MultiChoice | 10.5% | **13.6%**, 15.0% with the letter-bias correction | all 513 | chance 16.4%; both below chance, K2 less so |
| Omniscience stand-in | 51.8 | **53.1** | 266 | each with its own "I don't know" threshold from the practice set (Qwen 0.7, K2 0.9) |

**Index estimates** (0.25 × the four parts; the Omniscience part is a public stand-in):

| Debussy variant | When2Call | WildGuard | HLE | Omniscience | **Index** |
|---|---|---|---|---|---|
| Qwen3.5-4B (light default) | 93.2 (600 q) | 89.9 (880 rows) | 10.5 | 51.8 | **≈ 61.4** |
| K2-Horizon-7B | 89.3 | 93.6 (500 rows) | 15.0 | 53.1 | **≈ 62.8** |
| **Two specialists (recommended):** Qwen for tool calls, K2 for safety + knowledge | 93.2 | 93.6 | 15.0 | 53.1 | **≈ 63.7** |

## What we learned
- **K2 is the better judge and the more honest knower; Qwen is the better tool-caller.** K2's raw safety judgements are
  clearly sharper (harmful reply 0.87 vs 0.80, refusal 0.86 vs 0.81, before calibration, 300 test rows); Qwen's tool-call instincts are better, and more
  training pairs did not close K2's gap.
- **Calibration matters as much as the model.** With 150 calibration rows K2 merely tied Qwen on WildGuard (0.873 vs 0.871);
  with the same natural-mix rows Qwen got, K2 pulled ahead (0.928 vs 0.898), and with all 718 calibration rows and
  500 test rows, identical for both, the gap held (0.936 vs 0.905).
- **Re-checking old results pays off.** K2's 150 oldest calibration rows differed from a fresh run by 0.2-0.3 log-odds
  on average (no systematic shift; 2-3 of 60 yes/no answers flipped), most likely because they were saved before the
  double-BOS fix. They were re-run; K2's score did not move (0.928 → 0.927 on the same 300 test rows).
- **K2 is overconfident.** Its own "I don't know" threshold is 0.9 (Qwen: 0.7). With it, K2 answers only 9% of obscure
  questions and is honest on 98% of the rest.
- **Reading K2's internal state did not beat its own confidence (a negative result).** We trained a small "honesty
  detector" (logistic regression on K2's last hidden state at the answer point, plus its confidence) on the 400 practice
  questions to predict "this answer will be right". On the Omniscience stand-in it scored 51.8 against 53.1 for the
  plain confidence rule, and none of 8 variants beat the rule even on the practice set, so Debussy keeps the rule.
  The likely reason is that the practice questions (K2 right on 65%) look nothing like Omniscience's obscure ones
  (K2 answers 9%). Code: `scripts/run/k2_embed.py`, `scripts/score/honesty_probe.py`.
- **A second internal-state probe, for safety, also did not hold up (negative result).** A probe on K2's hidden state
  at the harmful-reply decision point (PCA + logistic regression, plus the usual log-odds) beat the current heads on
  nested cross-validation for all three safety questions (harmful reply 0.888 vs 0.846). On the 500 test rows it did
  not: harmful request 0.896 vs 0.929, refusal 0.962 vs 0.962, harmful reply 0.923 vs 0.917, average **0.927 vs 0.936**.
  It would also double the time per safety judgement (the hidden state needs a second read of the conversation). Debussy
  keeps the heads. Two lessons: a random-noise dry run showed that picking the best of 12 settings "wins" plain CV, so
  the rule was nested CV fixed in advance; and even nested CV inside the calibration set (balanced plus natural mix)
  did not predict the test set. Both internal-state probes looked better inside their training data and neither carried
  over. Code: `scripts/run/k2_wg_embed.py`, `scripts/score/wildguard_probe.py`.
- **K2 leans towards some answer letters, and correcting that helps a little.** On 600 HLE-shaped practice questions
  (MMLU-Pro, 5-10 options, correct letter uniform) K2 favours B and avoids the last option (5 options: B 0.27, E 0.12,
  fair share 0.20). A small per-option-count bias, chosen by nested CV on the practice set only (46.7% vs 45.8%), was
  then applied once to HLE: **13.6% → 15.0%** (about 7 more right answers of 513; on its own within noise, but in the
  direction the practice set predicted). It is part of `heads_k2-horizon.json` and applies to 5-10 options only.
  Code: `transfer_run.py hle_practice`, `scripts/score/hle_letter_bias.py`.
- **On the full WildGuardTest (all 1,725 rows) Debussy's safety judge is about level with WildGuard-7B**: average F1
  83.9 vs 84.3 (harmful request 83.9 vs 88.9, harmful reply 76.9 vs 75.4, refusal 91.0 vs 88.6), below GPT-4 (85.9).
  The annotator-agreed subset (0.936) is easier. Details: `WILDGUARD_FULL.md`.
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
python scripts/run/transfer_run.py wg_more_cal 0 150 && python scripts/run/transfer_run.py wg_more_cal 150 468   # 0-150: fresh re-run
python scripts/run/transfer_run.py wg_more_test 300 500
python scripts/score/compare_bases.py wg --write                # 718 calibration + 500 test rows, identical for both models
python scripts/score/wildguard.py data/k2-horizon/wg_all_718cal_500te.json --natural-weights --save --heads debussy/heads_k2-horizon.json
python scripts/run/transfer_run.py w2c 100 150 && python scripts/run/transfer_run.py w2c_more_train 100 250 && python scripts/run/transfer_run.py w2c_more_test 150 300
python scripts/checks/parity_check.py 5 k2-horizon server       # packaged Debussy = research pipeline, on K2
python -m debussy.model duo                                     # the two-specialist variant (Qwen in-process + K2 server)
```
