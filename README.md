<p align="center"><img src="brand/debussy-mark.svg" width="96" alt="Debussy logo"></p>

# Debussy

A fast **System 1 decision model** for the [Artificial Analysis Decision Index](https://artificialanalysis.ai/methodology/decision-index),
built on a 4-core CPU with no GPU and no API key. One small open model answers short yes/no questions in single passes
(no step-by-step reasoning), and tiny calibrated heads (`heads.json`) turn those answers into a probability for every
option. When unsure, Debussy prefers an honest "I don't know" over a guess.

**Estimated Decision Index ≈ 61.4** (When2Call 93.2 · WildGuardTest 0.899 · HLE 10.5 · Omniscience stand-in 51.8).
This is our own estimate, not an official score. Details, methods and caveats: **[docs/RESULTS.md](docs/RESULTS.md)**.

## Use it
```bash
pip install llama-cpp-python==0.3.36 numpy pandas pyarrow scikit-learn
mkdir -p models && curl -L -o models/Qwen3.5-4B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
python -m debussy.model    # three demo decisions (run from the repository root)
```
```python
from debussy import Debussy
d = Debussy()
d.when2call(tools, "Find me a flight to New York on March 14th", candidates)  # -> probability per reply type
d.wildguard(user_prompt, assistant_reply)        # -> P(harmful request), P(refusal), P(harmful reply)
d.choice(question, options + ["I don't know"])   # abstains when it is not confident
```

## Layout
```
debussy/                 the model (Python package)
  model.py                 Debussy class: when2call(), wildguard(), choice()
  engine.py                model-agnostic core: base-model profiles + backends (see docs/TRANSFER.md)
  heads.json               trained head weights and thresholds (tiny)
  paths.py                 where data/, models/ and heads.json live
  skills/                  when2call · wildguard · wildguard_rows · hle · omniscience · knowledge_proxy · retrieval
scripts/
  run/                     extra batches and running skills on a new base model (transfer_run.py)
  score/                   scoring and head fitting (choices made on training data only), verdict.py
  checks/                  parity_check.py (package = pipeline), engine_check.py (engine = saved results)
experiments/             early attempts: embedding classifier, probes, CPU LoRA, decomposed checks
docs/                    RESULTS.md (scores, methods, caveats) · TRANSFER.md (moving to a new base model)
brand/                   logo, wordmarks, exports; archive/ holds earlier logo versions
data/README.md           how to get the datasets (not stored here; some are gated)
```
Scripts in `scripts/` run from anywhere; package modules run as `python -m debussy.skills.<name>` from the repository root.

## Name
After the composer Claude Debussy: the logo is a *Clair de lune* crescent and an *Arabesque* curl, and the name is
a nod to Debussy being built with help from Claude.

Built by the project owner with help from Claude (Anthropic's AI assistant). Not an Anthropic model.
Third-party licences: [NOTICE.md](NOTICE.md).
