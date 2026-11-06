---
license: apache-2.0
language:
- en
pipeline_tag: text-classification
base_model:
- unsloth/Qwen3.5-4B-GGUF
- IFM/K2-Horizon-7B-GGUF
datasets:
- nvidia/When2Call
- allenai/wildguardmix
- TIGER-Lab/MMLU-Pro
- ArtificialAnalysis/AA-Omniscience-Public-v1.1
tags:
- decision-model
- safety
- guardrail
- tool-calling
- calibration
- abstention
- gguf
- cpu
---

<!-- upload brand/debussy-logo.svg from the GitHub repository next to this card -->
<p align="center"><img src="debussy-logo.svg" width="360" alt="Debussy logo"></p>

# Debussy

**A fast System 1 decision model, built on a 4-core CPU with no GPU and no API key.**

Debussy makes the quick judgements an AI agent needs, without step-by-step reasoning. Each call returns a
probability for every option:

- **Safety:** is the user's request harmful? Did the assistant refuse? Is the assistant's reply harmful?
- **Tool calls:** given the tools and the request, should the agent call the tool, ask for missing details, or say it
  cannot help?
- **Knowledge questions:** which option is right, or "I don't know" when it is not sure. Debussy prefers an honest
  "I don't know" to a confident guess.

It follows the decision-model format of the [Artificial Analysis Decision Index](https://artificialanalysis.ai/methodology/decision-index):
one pass per decision, a probability per option, no free-form text.

By **Debussy Labs** ([debussylabs.com](https://debussylabs.com)). Built with help from Claude, Anthropic's AI
assistant. Not an Anthropic model.

## Variants

| Variant | Base model(s) | RAM | Estimated index |
|---|---|---|---|
| **`DebussyDuo()`** (recommended) | Qwen3.5-4B for tool calls, K2-Horizon-7B for safety and knowledge | ~9 GB | **≈ 63.4** |
| `Debussy(profile="k2-horizon", backend="server")` | K2-Horizon-7B only, run through `llama-server` | ~6 GB | ≈ 62.4 |
| `Debussy()` (light) | Qwen3.5-4B only, in-process | ~3 GB | ≈ 61.4 |

Both base models are open (Apache-2.0) and used as 4-bit GGUF files. K2-Horizon is a reasoning model; Debussy runs it
with its thinking switched off, so every decision is still a single pass. Debussy does not change the base models'
weights: what it adds is the questions it asks and small calibrated "heads" (a few numbers each) in `heads.json` and
`heads_k2-horizon.json`.

## Results

Our own measurements on test questions that were never used to fit or choose anything. **These are estimates, not
official Artificial Analysis scores.**

| Benchmark (25% of the index each) | Debussy Duo | Debussy light (Qwen only) | Test set |
|---|---|---|---|
| When2Call (accuracy) | 93.2% | 93.2% | 600 test questions |
| WildGuardTest (average F1) | **0.936** | 0.899 | annotator-agreed rows: 500 (Duo), 880 (light) |
| HLE-MultiChoice (accuracy) | 13.6% | 10.5% | all 513 text-only multiple-choice questions |
| AA-Omniscience (½ accuracy + ½ non-hallucination) | 53.1 | 51.8 | stand-in: AA's 266 public questions |
| **Index estimate** | **≈ 63.4** | **≈ 61.4** | |

WildGuardTest for the Duo, per question: harmful request 0.929, refusal 0.962, harmful reply 0.917. On the same 500
rows with the same calibration rows, the Qwen-only judge scores 0.905.

**Read these numbers with care:**
- **WildGuardTest** here uses only the rows where all three human annotators agree, as Artificial Analysis does. That
  subset is easier than the full test set that guard models usually report on, so these F1 scores are not comparable
  with published guard-model results.
- **AA-Omniscience** questions are private; our number is on Artificial Analysis's public sample, with wrong options
  written by a model.
- **HLE-MultiChoice:** both variants are below chance (16.4%). A System 1 model does not solve these questions; it
  mostly needs to avoid confident wrong answers elsewhere.
- Test sets are 300-600 questions, so differences of 2-3 points can be noise.

## Quick start

```bash
git clone https://github.com/abutauskas/debussy && cd debussy
pip install llama-cpp-python==0.3.36 numpy pandas pyarrow scikit-learn
mkdir -p models && curl -L -o models/Qwen3.5-4B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
python -m debussy.model          # three demo decisions with the light variant
```

```python
from debussy import Debussy, DebussyDuo
d = Debussy()                    # or DebussyDuo() with K2-Horizon served by llama-server
d.wildguard(user_prompt, assistant_reply)
# -> e.g. {"prompt_harm": 0.998, "refusal": 1.0, "response_harm": 0.003}
d.when2call(tools, "Find me a one-way flight to New York on March 14th.", candidates)
# -> e.g. {"tool_call": 0.011, "request_for_info": 0.971, "cannot_answer": 0.018, "direct": 0.0}
d.choice("What is the capital of Australia?", ["Sydney", "Canberra", "Melbourne", "Perth", "I don't know"])
# -> e.g. {"Canberra": 1.0, ..., "I don't know": 0.0}
```

The K2-Horizon variants need a current llama.cpp build of `llama-server` (the pinned `llama-cpp-python` cannot load
K2-Horizon). Setup and the full comparison: `docs/K2_EXPERIMENT.md` in the repository.

## How it works

1. **One read, short questions.** The base model reads the input once, then answers a few yes/no questions (for
   safety, three questions in one chained pass). Debussy reads the model's probability of "yes" against "no" rather
   than generating text.
2. **Calibrated heads.** Tiny logistic regressions turn those probabilities into calibrated decisions. Which head to use
   is chosen by cross-validation on training rows only.
3. **Honest abstention.** For knowledge questions the "I don't know" option is not shown to the model. Debussy answers
   only when its confidence in the best option clears a threshold set on a separate practice set, and says "I don't
   know" otherwise. On the obscure Omniscience questions K2-Horizon answers only 9% of them, and when it does not know
   the answer it says "I don't know" 98% of the time instead of guessing wrong.

## Training and calibration data

The base models are not fine-tuned. The heads were fitted on training splits only:
- **Safety:** 718 rows from the WildGuardMix training split (468 balanced over label combinations, 250 at the
  natural label mix).
- **Tool calls:** When2Call training pairs (799 for Qwen, 500 for K2-Horizon).
- **"I don't know" threshold:** 400 MMLU-Pro questions turned into 4-option questions.

No test question was used to fit or choose anything. Humanity's Last Exam was used for evaluation only.

## Intended use

- A fast, local first-pass judge inside agent pipelines: flagging possibly harmful requests or replies, and routing
  tool calls.
- Research on calibration, abstention and small decision models.

**Out of scope:**
- **Being the only safety layer.** Use Debussy as one signal, with human review for anything high-stakes.
- **Languages other than English.** It was calibrated and tested on English only.
- **Long conversations.** The safety judge reads only the first 600 tokens of the request and of the reply.
- **Hard reasoning.** It is a System 1 model by design.
- **Generating text.** It returns probabilities, not answers in prose.

## Limitations

- **Speed on small CPUs.** On the 4-core cloud CPU it was built on, a safety judgement with K2-Horizon takes about
  13 s, a tool-call decision with Qwen about 6 s and a knowledge question about 3 s. GPUs and laptops with fast
  matrix units should be much faster, but we have not measured that yet.
- **What counts as harm.** The safety judge learns WildGuardMix's definitions of harm and refusal. Other policies or
  domains may need re-calibration (a few hundred labelled rows are enough to re-fit the heads).
- **Overconfidence.** K2-Horizon is overconfident before calibration; its "I don't know" threshold is 0.9 (Qwen: 0.7).

## What did not work

We report these on purpose:
- A probe on K2-Horizon's internal state to predict "this answer will be right" scored worse than its plain
  confidence (51.8 against 53.1 on the Omniscience stand-in).
- Combining both models' knowledge answers, and Wikipedia retrieval for obscure questions, gave little or nothing.

## Licence and attribution

Debussy's code and heads are released under Apache-2.0. The base models are Apache-2.0 (Qwen3.5-4B by the Qwen team,
K2-Horizon-7B by the Institute of Foundation Models). Inference runs on llama.cpp (MIT). Calibration and evaluation
used When2Call (NVIDIA), WildGuardMix (Ai2, gated), MMLU-Pro (TIGER-Lab), Humanity's Last Exam (CAIS and Scale AI,
gated, evaluation only) and AA-Omniscience-Public (Artificial Analysis); see each dataset's own terms. No dataset is
redistributed here.

**Name:** after the composer Claude Debussy. The logo is a *Clair de lune* crescent and an *Arabesque* curl, and the
name is a nod to Debussy being built with help from Claude.

**Version:** 0.1 (November 2026).
