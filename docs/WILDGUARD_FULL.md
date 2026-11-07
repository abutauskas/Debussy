# Debussy on the full WildGuardTest

Our headline safety number (F1 0.936) uses only the 880 WildGuardTest rows where all three annotators agree, which is
what the Artificial Analysis Decision Index uses. That subset is easier, and it is not what guard models report. So we
scored Debussy's safety judge on **all 1,725 rows** as well.

**Setup:** Debussy Duo's safety judge (K2-Horizon-7B, thinking off) with its **saved heads**, fitted on 718 calibration
rows from WildGuardMix's *training* split. Nothing was fitted or chosen on the test set. Each question is scored on the
rows that have its label (1,699 / 1,720 / 1,709). Other models' numbers are as published in the WildGuard paper
(Han et al., 2024, arXiv 2406.18495, Table 4, "Total" columns); we did not re-run them.

| Full WildGuardTest, F1 | Harmful request | Harmful reply | Refusal | Average |
|---|---|---|---|---|
| **Debussy Duo** | 83.9 | **76.9** | **91.0** | **83.9** |
| WildGuard-7B | **88.9** | 75.4 | 88.6 | 84.3 |
| GPT-4 | 87.9 | 77.3 | 92.4 | 85.9 |
| Llama-Guard2 | 70.9 | 66.5 | 53.8 | 63.7 |

## What it means
- **About level with WildGuard-7B overall** (83.9 vs 84.3): better at harmful replies (+1.5) and refusals (+2.4), worse
  at harmful requests (-5.0). WildGuard was trained on about 87,000 WildGuardMix examples; Debussy's heads were fitted on
  718 of them, with the base model unchanged.
- **Below GPT-4** (85.9 average).
- **The annotator-agreed subset flatters every model.** On it Debussy scores 0.936; on the full set, 0.839. Compare
  guard models only on the same rows.
- Fairness: WildGuard and Debussy were both trained or calibrated on WildGuardMix's training split, so this test is
  in-distribution for both; for Llama-Guard2 and GPT-4 it is not.

## Caveats
- One run; published numbers come from the paper, and its handling of rows without a label may differ slightly.
- English only; the safety judge reads the first 600 tokens of the request and of the reply.

Reproduce: `transfer_run.py wg_full 0 1225` (K2-Horizon via llama-server), then
`python scripts/score/wildguard_full.py k2-horizon`.
