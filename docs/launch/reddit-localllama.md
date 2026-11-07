**Title:** Debussy: a calibrated "System 1" decision model for agents that runs on CPU with llama.cpp (Qwen3.5-4B + K2-Horizon-7B with thinking off)

**Body:**

I've been working on a small decision model for agents and wanted to share it here, because it's built entirely on
GGUF + llama.cpp and was developed on a 4-core CPU.

**What it does:** three kinds of decision, each in a single pass with a probability per option:
- safety: harmful request? did the assistant refuse? harmful reply?
- tool calls: make the call, ask for missing details, or say it can't help
- knowledge: pick an option, or "I don't know" when it isn't confident

**How:** the base model reads the input once and answers short yes/no questions; Debussy reads P(yes) vs P(no) from
the logits instead of generating, then tiny logistic heads calibrate the decisions. No fine-tuning. For K2-Horizon-7B
(a reasoning model) the thinking block is opened and closed immediately, so it's a single pass.

**The interesting finding:** two small models are good at different decisions. Qwen3.5-4B is the better tool-caller
(94.0% vs 89.3% on the same When2Call rows); K2-Horizon-7B is the sharper safety judge (F1 0.936 vs 0.905 on the same
500 WildGuardTest rows). So the recommended "Duo" routes each decision to its specialist (~9 GB RAM total).

**Numbers** (our estimates on the Artificial Analysis Decision Index parts, not official): When2Call 93.2%, WildGuardTest
0.936 (annotator-agreed rows only), Omniscience stand-in 53.1, HLE-MC 15.0% (below chance). Estimated index ≈ 63.7.
On the **full** WildGuardTest (1,725 rows) the safety judge averages F1 83.9 vs WildGuard-7B's published 84.3:
better on harmful replies (76.9 vs 75.4) and refusals (91.0 vs 88.6), worse on harmful requests (83.9 vs 88.9).

**Speed on a 4-core cloud CPU:** a safety judgement ~13 s with K2 (Q4_K_M), a tool-call decision ~6 s with Qwen 4B,
a knowledge question ~1–3 s. Haven't measured GPUs yet.

**Small gotchas we hit:** a double BOS token silently skewed K2's early results; K2's tokenizer puts a space before the
answer letter (" B"), so reading the letter probabilities naively gave a flat 0.25; llama-server's default prompt cache
ate 8 GB of RAM until `--cache-ram 0`; and K2 leans towards answer "B" and away from the last option, which a small
bias correction (fit on practice questions only) fixed: HLE 13.6% → 15.0%.

Apache-2.0, code + heads + write-ups including what didn't work: https://github.com/abutauskas/Debussy
Model card: https://huggingface.co/debussylabs/debussy · Site: https://debussylabs.com

Built with help from Claude (not an Anthropic model). Feedback very welcome, especially on GPU speeds if anyone tries it.
