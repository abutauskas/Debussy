<p align="center"><img src="brand/debussy-logo.svg" width="400" alt="Debussy logo"></p>

A fast **System 1 decision model** for the [Artificial Analysis Decision Index](https://artificialanalysis.ai/methodology/decision-index),
built on a 4-core CPU with no GPU and no API key. Small open models answer short yes/no questions in single passes
(no step-by-step reasoning), and tiny calibrated heads (`heads.json`) turn those answers into a probability for every
option. When unsure, Debussy prefers an honest "I don't know" over a guess.

**Estimated Decision Index ≈ 63.4** with two specialists (`DebussyDuo`: Qwen3.5-4B for tool calls, K2-Horizon-7B for
safety and knowledge): When2Call 93.2 · WildGuardTest 0.936 · HLE 13.6 · Omniscience stand-in 53.1. The light
single-model default (`Debussy()`, Qwen3.5-4B only) is ≈ 61.4. These are our own estimates, not official scores. Details, methods and caveats: **[docs/RESULTS.md](docs/RESULTS.md)**.

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

### Variants
| Variant | Base model(s) | RAM | Estimated index |
|---|---|---|---|
| `Debussy()` (default) | Qwen3.5-4B, in-process | ~3 GB | ≈ 61.4 |
| `Debussy(profile="k2-horizon", backend="server")` | K2-Horizon-7B with thinking off, via `llama-server` | ~6 GB | ≈ 62.4 |
| **`DebussyDuo()`** (recommended) | Qwen for tool calls, K2 for safety and knowledge | ~9 GB | **≈ 63.4** |

Every variant still makes each decision in a single pass. K2 needs a current llama.cpp build of `llama-server`
(the pinned `llama-cpp-python` cannot load it). How it compares, and how to run it: **[docs/K2_EXPERIMENT.md](docs/K2_EXPERIMENT.md)**.

## Layout
```
debussy/                 the model (Python package)
  model.py                 Debussy and DebussyDuo: when2call(), wildguard(), choice()
  engine.py                model-agnostic core: base-model profiles + backends (see docs/TRANSFER.md)
  heads.json               trained head weights and thresholds (tiny); heads_k2-horizon.json for K2
  paths.py                 where data/, models/ and heads.json live
  skills/                  when2call · wildguard · wildguard_rows · hle · omniscience · knowledge_proxy · retrieval
scripts/
  run/                     extra batches and running skills on a new base model (transfer_run.py)
  score/                   scoring and head fitting (choices made on training data only), verdict.py
  checks/                  parity_check.py (package = pipeline), engine_check.py (engine = saved results)
experiments/             early attempts: embedding classifier, probes, CPU LoRA, decomposed checks
docs/                    RESULTS.md (scores, methods, caveats) · TRANSFER.md (moving to a new base model)
                         · K2_EXPERIMENT.md (Qwen vs K2-Horizon, the two-specialist variant)
brand/                   logo, wordmarks, exports; archive/ holds earlier logo versions
hf/README.md             model card for Hugging Face (upload with the heads files and the logo)
data/README.md           how to get the datasets (not stored here; some are gated)
```
Scripts in `scripts/` run from anywhere; package modules run as `python -m debussy.skills.<name>` from the repository root.

## Name
After the composer Claude Debussy: the logo is a *Clair de lune* crescent and an *Arabesque* curl, and the name is
a nod to Debussy being built with help from Claude.

Built by the project owner with help from Claude (Anthropic's AI assistant). Not an Anthropic model.
Licence: [Apache-2.0](LICENSE). Third-party licences: [NOTICE.md](NOTICE.md).
