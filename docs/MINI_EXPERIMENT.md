# Debussy Mini (Qwen3.5-2B): a negative result

**Question:** would a smaller base model, Qwen3.5-2B (1.3 GB, half the size of Debussy Light's Qwen3.5-4B), make a
useful "Debussy Mini": fast enough for free CPU hosting, without losing much quality?

**How:** the same skills, prompts and rules as the other variants (`qwen3.5-2b` profile in `debussy/engine.py`), run with
`transfer_run.py` in-process on the same 4-core CPU. Safety heads fitted on the same 718 calibration rows (training
split) and scored on the same 500 annotator-agreed test rows as Light and K2; the row labels were checked to be
identical.

## Results

| WildGuardTest F1 (same 500 test rows) | **Mini (2B)** | Light (4B) | K2 (7B) |
|---|---|---|---|
| Harmful request | 0.731 | 0.905 | 0.929 |
| Refusal | 0.890 | 0.912 | 0.962 |
| Harmful reply | **0.527** | 0.897 | 0.917 |
| **Average** | **0.716** | 0.905 | 0.936 |

| Speed on this 4-core CPU (same package) | Mini | Light |
|---|---|---|
| Safety judgement | 4.6 s | 5.3–6.4 s (demo runs) |
| Tool-call decision | 4.4 s | about 6 s |

## Verdict

**Not worth shipping.** Halving the base model costs 19 F1 points on safety (and 37 on spotting a harmful reply) for
only about 20–30% more speed, because most of the time goes into reading the input, which shrinks less than the model.
The tool-call and knowledge runs were stopped early once that was clear, so there are no Mini numbers for them; the
CPU went to the full WildGuardTest instead.

**What this tells us:** for these judgements, model size below 4B hurts far more than it helps. Speed should come
from hardware (a GPU) or shorter inputs, not a smaller base model.

Reproduce: `DEBUSSY_PROFILE=qwen3.5-2b python scripts/run/transfer_run.py wg 468 500` (+ `wg_natural 170 1`,
`wg_natural 80 2`), then `python scripts/score/wildguard.py data/qwen3.5-2b/wg_all_718cal_500te.json --natural-weights`.
