# Debussy: a System 1 decision model on a 4-core CPU

**Goal:** score as high as possible on the [Artificial Analysis Decision Index](https://artificialanalysis.ai/methodology/decision-index)
using minimal compute: no GPU, no API key, no paid services.

**What it is:** one small open model (Qwen3.5-4B, 4-bit, 2.7 GB, runs in llama.cpp) answering short yes/no
questions in single passes, with no step-by-step reasoning. Tiny calibrated "heads" (a few numbers each, fit on public
TRAINING data only) turn those answers into a probability for every option, which is the decision-model format
Artificial Analysis uses.

## Results (test questions the system never trained on)

| Benchmark (25% of index each) | Start of project | **Debussy** | Test size | Leaders* |
|---|---|---|---|---|
| When2Call (accuracy) | 38% (0.8B zero-shot) | **93.2%** model-only, 94.3% with tool-list check | 600 | 84 / 92.6 |
| WildGuardTest (avg F1, annotator-agreed rows) | 0.69 (0.8B probe, all rows) | **0.899** | 880 (all agreed rows) | ~95 |
| AA-Omniscience-Decision (½ accuracy + ½ non-hallucination) | n/a | **51.8** on AA's public sample (stand-in; real set is private) | 266 | ~35-50 |
| HLE-MultiChoice (accuracy) | n/a | **10.5%** (chance 16.4%) | all 513 (same set as AA) | 21-24 |

\*Leader numbers from the Artificial Analysis article (Oct 2026) and the BenchLM snapshot the user shared; the
leaderboard changes over time.

**Index estimate:** 0.25 × (93.2 + 89.9 + 10.5 + 51.8) ≈ **61.4**. For comparison, Artificial Analysis's launch article
lists the best decision model at 58 and Claude Haiku 5.5 (Low) at 61; a later snapshot shows ~65 at the top. This is
an **estimate, not a measured leaderboard score**: Artificial Analysis runs its own harness, and the Omniscience
questions are private. This is an
**estimate, not a measured leaderboard score**. See the caveats below.

## How each part works

**When2Call.** Artificial Analysis shows the model the tools, the request and four candidate replies. Debussy reads
the tools and request once, saves the model state, then asks five short yes/no questions from that state:
1. *Is the function in the proposed tool call one of the available tools?* If no, it chooses **cannot answer**.
2. *Is every argument value in the proposed call stated in the request (not made up)?*
3. *Does the request give a value for every **required** parameter of that tool?*
4. *Has the user already given the info the "ask" reply asks for?*
5. *Is the info the "ask" reply wants actually **required**, or optional?*

A small logistic head (fit on 799 When2Call TRAIN pairs; cross-validation kept checks 2, 3 and 5) turns them into **call the tool** vs **ask for info**.
History: 38% zero-shot → 58% embedding classifier → 63% 0.8B probe → 59% LoRA fine-tune (CPU) → 89% (checks 1, 2, 4)
→ 91.8% (all five, 300 training pairs) → 92.9% (500 pairs, 425 test questions) → 92.7% (500 pairs, 600 test questions)
→ 92.8% (799 pairs, 600 test questions) → **93.2%** with the tool-exists cut-off chosen on training data (-0.25: the
level 99% of real training tools pass, instead of a fixed 0). Gains from more pairs have flattened out: the head has learned what these
five checks can tell it. Remaining errors are almost all tool-call ↔ ask-for-info confusions.

**WildGuardTest.** One pass reads the conversation plus three questions (harmful request? / refusal? /
harmful reply?) and answers them in turn. A small calibrated head per question turns the three answers into
decisions; the head type (none / threshold shift / logistic / logistic with interaction terms such as
P(harmful request)·P(not refused)) is chosen by cross-validation on 718 WildGuard TRAIN rows (harmful request: none;
refusal and harmful reply: the interaction head). Scores on all **880** annotator-agreed test rows: harmful request
0.917, refusal 0.900, **harmful reply 0.880** → **average 0.899**.

The biggest fix (Debussy round 2) was a **calibration-mix mismatch**: the first 468 calibration rows were drawn
evenly over the 6 label combinations, which made rare cases like "harmless request, harmful reply" 58× too common
(5.8% vs 0.1% in the training split). Re-weighting them to the training split's natural mix (training statistics
only) let cross-validation pick the interaction head, lifting harmful-reply F1 from 0.805 to 0.878 on the first 500
test rows. On the full 880 rows: previous settings 0.886 → current **0.898**.

Other things tried: a longer context limit (1,500 instead of 600 tokens, re-running the 192 clipped rows) — no gain
on training cross-validation (0.883 vs 0.887), so kept 600 (2× cheaper on long inputs); long conversations are simply
harder, not cut off. Adding 170 naturally-drawn calibration rows (468 → 638) changed test F1 by −0.007 on 500 rows
(noise); we kept the larger set by our pre-set rule "fit on more data", not "pick whichever scores best on test".
A further 80 natural rows (718 total) gave 0.899 on all 880 test rows (vs 0.898): calibration has saturated.
Tried and rejected: a "v2" prompt (judge reply harm before refusal, list misinformation explicitly). It was worse on
TRAIN cross-validation (harmful reply 0.727 vs 0.805) and on the first 80 test rows (avg 0.779 vs 0.832), so v1 stays.

**"I don't know" policy (AA-Omniscience-Decision).** Non-hallucination = 100 × (1 − wrong / (wrong + "I don't know")).
Debussy answers only when its top option has ≥ 70% probability, otherwise "I don't know". On 400 4-option
knowledge questions built from MMLU-Pro (threshold chosen on one half, scored on the other):
always answer 34 · always "I don't know" 50 · **threshold 0.7 → 66** · hardest subjects only → 56.
**Closer stand-in: Artificial Analysis's own public sample** (`AA-Omniscience-Public-v1.1`, 300 open-answer questions,
Apache-2.0). The Decision version needs 4 options, so the model wrote 3 plausible wrong answers per question (it saw
the right answer while doing so); 266 questions got 3 usable options. A separate pass, without the answer, scored
the options, and the **pre-set 0.7 rule was applied unchanged**: always answer 23.3 · always "I don't know" 50.0 ·
**Debussy 51.8** (answers 15%, right on most of those; non-hallucination 94). On genuinely obscure facts the model
is rarely confident, so the honest score is ~52, not the 66 the easier MMLU-Pro stand-in suggested. Caveat: wrong
options written by the same model are sometimes longer than the right answer, which can be a giveaway.

Tried: showing "E. I don't know" to the model and using its own probability for E. Worse: the best E-based rule
scored ~59 on the calibration half, and the E option also made the A-D answers slightly less sharp (held-out 63.6
vs 66.3). So Debussy hides any "I don't know" option from the model and abstains on low confidence instead.

## Caveats (read before quoting any number)
- **Not an official score.** Artificial Analysis runs its own prompts and harness, and some details are unpublished.
  Our protocol follows their methodology page where it is public.
- **Sample sizes** (CPU budget): When2Call 600 of 3,652 test questions (±~2 points, 95% CI); WildGuard uses all 880 agreed rows (±~2).
- **When2Call tool-list quirk:** in this test set every "cannot answer" item's proposed call names a tool that is
  not in the tool list. "Never call a tool you don't have" is a legitimate rule, but it is also a dataset artifact,
  so the headline uses the **model-only** score (93.2%); the string check (94.3%) is opt-in. Artificial Analysis also
  drops 229 ambiguous "ask for info" items; we could not identify them, so our test set includes them.
- **WildGuard:** we used the 880 rows where all annotators agreed (AA uses 835 of them); responses and prompts are
  clipped to 600 tokens each.
- **Omniscience:** the real 500 questions are private; the headline uses AA's public sample with model-written wrong
  options (MMLU-Pro, which is easier, gave 66).
- **HLE:** measured on all 513 text-only multiple-choice questions (the same subset AA uses), letter scoring, one
  pass each; one 13.5k-token question was cut to its last 8,000 tokens (the answer choices are at the end).
  10.5% is below the 16.4% chance level: HLE keeps questions that frontier models get wrong, so the most tempting
  answer is often the trap. We report it as measured; we did not tune anything against HLE.

## Files
- `debussy/model.py` (`from debussy import Debussy`): the model package (`Debussy().when2call(...)`, `.wildguard(...)`, `.choice(...)`)
- `debussy/heads.json`: all trained head weights and thresholds (tiny)
- `models/Qwen3.5-4B-Q4_K_M.gguf`: base model (download: `unsloth/Qwen3.5-4B-GGUF`)
- `debussy/skills/`: when2call, wildguard, knowledge_proxy, hle, omniscience, retrieval (each also runs with `python -m debussy.skills.<name>`)
- `scripts/score/`: scoring and head fitting (choices made on TRAIN only); `scripts/run/`: the extra batches
- `scripts/checks/parity_check.py`: confirms the packaged model makes the same decisions as the research pipeline (15/15 and 45/45)
- `brand/`: the Debussy logo (arabesque note with crescent-moon head), wordmarks and exports
- `data/*.json`: raw per-question model outputs behind every number above

## Reproduce
```bash
pip install llama-cpp-python==0.3.36 numpy pandas pyarrow scikit-learn
# data: nvidia/When2Call (public), allenai/wildguardmix (gated), TIGER-Lab/MMLU-Pro (public) into data/
python -m debussy.skills.when2call models/Qwen3.5-4B-Q4_K_M.gguf 150 600 data/w2o2_4b.json && python scripts/score/when2call.py data/w2o2_4b.json
#   (we scored test items 0-174 there and 175-424 with: python scripts/run/when2call_more_test.py MODEL 175 425 data/w2o2_4b_extra.json;
#    merged file: data/w2o2_4b_all.json; +200 training pairs: python scripts/run/when2call_more_train.py MODEL 100 data/w2c_more_train.json
#    final merged file (500 train pairs + 425 test): data/w2o2_4b_all_500tr.json)
python -m debussy.skills.wildguard models/Qwen3.5-4B-Q4_K_M.gguf 120 300 data/wg1p_4b.json
python scripts/run/wildguard_more_train.py models/Qwen3.5-4B-Q4_K_M.gguf data/wg1p_4b.json 200 data/wg_more_train.json   # +178 calibration rows
python scripts/run/wildguard_more_test.py models/Qwen3.5-4B-Q4_K_M.gguf 300 500 data/wg_more_test.json                   # +200 test rows
WG_SEED=2 python scripts/run/wildguard_more_train.py models/Qwen3.5-4B-Q4_K_M.gguf data/wg1p_4b.json 200 data/wg_more_train2.json  # +170 more
WG_CLIP=1500 python scripts/run/wildguard_reclip.py data/wg1p_4b_all_468tr.json data/wg_all_clip1500.json   # longer-context test (rejected)
python scripts/run/wildguard_natural_train.py 170 data/wg_natural_train.json                               # +170 natural calibration rows
python scripts/run/wildguard_more_test.py models/Qwen3.5-4B-Q4_K_M.gguf 500 880 data/wg_more_test2.json      # remaining agreed test rows
python scripts/score/wildguard.py data/wg_all_638tr_880te.json --natural-weights --save   # 638 calibration + 880 test rows
python -m debussy.skills.knowledge_proxy models/Qwen3.5-4B-Q4_K_M.gguf 400 data/omni_4b.json && python -m debussy.skills.knowledge_proxy --score data/omni_4b.json
python -m debussy.skills.hle data/hle_4b.json && python -m debussy.skills.hle --score data/hle_4b.json     # needs approved access to cais/hle
python -m debussy.skills.omniscience gen data/omni_public_gen.json && python -m debussy.skills.omniscience eval data/omni_public_gen.json data/omni_public_eval.json
python -m debussy.skills.omniscience --score data/omni_public_eval.json
python scripts/checks/parity_check.py 15
```
Speed on 4 CPU cores: When2Call ~14 s per decision, WildGuard ~15 s, multiple choice ~3.7 s. A GPU would make these well under a second.

## What would move the score next
1. **HLE (10.5%).** The weakest part, and hard for any small model: it rewards deep expert knowledge. A bigger base
   model is the main lever; we deliberately did not try tricks that exploit how HLE was built.
2. **WildGuard (0.898 vs ~0.95).** Longer context and more calibration rows have been tried (little gain); the next
   step is a short LoRA fine-tune of the 4B on WildGuardMix TRAIN (needs a GPU; 1-2 rented GPU hours, roughly $5-20).
3. **Tool-call vs ask-for-info (When2Call, ~24 errors / 425).** Most remaining errors are borderline "is this detail
   required?" cases; more training pairs or a fine-tune would help.
4. **Speed.** On a GPU every decision would take well under a second instead of 4-15 s.

## Timeline of this run (CPU only)
- Day 1: embedding classifier (58%), 0.8B probe (63%), CPU LoRA fine-tune (59%, worse), WildGuard 0.8B probe (0.69, other session).
- Day 2 (12-hour improvement run): llama.cpp (2× faster), option-aware When2Call judge (89% → 92.9%),
  single-pass WildGuard judge (0.69 → 0.885 on 500 rows), "I don't know" policy (proxy 66), packaging.
- Day 3 (8-hour round, renamed Debussy): WildGuard calibration-mix fix + interaction heads (0.886 → 0.898 on all 880
  agreed rows); longer context tested (no gain); When2Call richer heads tested (no gain); explicit "I don't know" option tested;
  more training data (When2Call 799 pairs, WildGuard 718 rows: both saturated); **HLE measured (10.5%)**; Omniscience checked on
  AA's public sample (51.8); packaged model verified against the pipeline; logo designed.
