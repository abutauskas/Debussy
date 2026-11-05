<p align="center"><img src="brand/debussy-mark.svg" width="96" alt="Debussy logo"></p>

# Debussy

A fast **System 1 decision model** for the [Artificial Analysis Decision Index](https://artificialanalysis.ai/methodology/decision-index),
built on a 4-core CPU with no GPU and no API key. One small open model answers short yes/no questions in single passes
(no step-by-step reasoning), and tiny calibrated heads (`heads.json`) turn those answers into a probability for every
option. When unsure, Debussy prefers an honest "I don't know" over a guess.

**Estimated Decision Index ≈ 61** (When2Call 92.8 · WildGuardTest 0.899 · HLE 10.5 · Omniscience stand-in 51.8).
This is our own estimate, not an official score. Details, methods and caveats: **[RESULTS.md](RESULTS.md)**.

## Use it
```bash
pip install llama-cpp-python==0.3.36 numpy pandas pyarrow scikit-learn
mkdir -p models && curl -L -o models/Qwen3.5-4B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
python debussy.py          # three demo decisions
```
```python
from debussy import Debussy
d = Debussy()
d.when2call(tools, "Find me a flight to New York on March 14th", candidates)  # -> probability per reply type
d.wildguard(user_prompt, assistant_reply)        # -> P(harmful request), P(refusal), P(harmful reply)
d.choice(question, options + ["I don't know"])   # abstains when it is not confident
```

## Layout
- `debussy.py`, `heads.json`: the packaged model
- `engine.py`, `TRANSFER.md`: the model-agnostic core; Debussy's skills run on any base model through a profile
  (Qwen3.5 and K2-Horizon so far)
- `w2c_*.py`, `wildguard_yesno.py`, `wg_*.py`, `omni_*.py`, `hle_eval.py`: running each benchmark skill
- `score_*.py`, `verdict.py`, `parity_check.py`, `engine_check.py`: scoring, head fitting (train-only choices) and checks
- `transfer_run.py`: run every skill on a new base model
- `lm_decide.py`, `when2call_head.py`, `extract_lm_feats.py`, `train_lora.py`, `decompose.py`: early experiments
- `brand/`: logo (an arabesque note with a crescent-moon head), wordmarks, exports
- `data/README.md`: how to get the datasets (not stored here; some are gated)

## Name
After the composer Claude Debussy: the logo is a *Clair de lune* crescent and an *Arabesque* curl, and the name is
a nod to Debussy being built with help from Claude.

Built by the project owner with help from Claude (Anthropic's AI assistant). Not an Anthropic model.
Third-party licences: [NOTICE.md](NOTICE.md).
