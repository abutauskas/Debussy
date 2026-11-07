**Title:** Show HN: Debussy – an open System 1 decision model for AI agents, built on a 4-core CPU

**URL:** https://debussylabs.com

**Text:**

Hi HN. Debussy is a small, open model for the quick decisions an AI agent makes all day: is this request harmful
(and did the assistant refuse, and is the reply harmful), should the agent call this tool or ask for missing details
first, and does it actually know the answer to a question or should it say "I don't know".

It's deliberately "System 1": no chain of thought, no generation. A small open model (Qwen3.5-4B, plus K2-Horizon-7B
with its thinking switched off) reads the input once and answers a few yes/no questions; Debussy reads the yes/no
probabilities and turns them into calibrated decisions with tiny logistic heads (a few hundred numbers in total). The
base models' weights are unchanged. Everything ran on a 4-core cloud CPU: no GPU, no API key.

Our own estimates on the four parts of the Artificial Analysis Decision Index (not official scores):
- When2Call (tool calls): 93.2%
- WildGuardTest (safety): F1 0.936 on the rows where all annotators agree (the subset Artificial Analysis uses). On the
  full 1,725-row test set that guard models report on, it averages 83.9, about level with WildGuard-7B's published
  84.3 (better on harmful replies and refusals, worse on harmful requests) and below GPT-4's 85.9.
- Omniscience stand-in (knowledge with "I don't know"): 53.1. On obscure questions it answers only 9% of the time,
  and when it doesn't know it says so 98% of the time instead of guessing.
- HLE-MultiChoice: 15.0%, below chance (16.4%). A System 1 model doesn't solve these, and we don't pretend otherwise.

Things that didn't work, which we've written up: two probes on the model's internal state (one looked great inside the
calibration data and lost on test), combining the two models' knowledge answers, and Wikipedia retrieval. A dry run
with random noise showed that picking the best of 12 probe settings "wins" ordinary cross-validation, so new ideas are
now judged with nested cross-validation fixed before looking at test data.

Caveats: about 13 s per safety judgement on that 4-core CPU (GPU numbers to come), English only, and it's one signal
for a safety system, not a complete one.

Code, heads and write-ups (Apache-2.0): https://github.com/abutauskas/Debussy · model card:
https://huggingface.co/debussylabs/debussy

Full disclosure: I built this with a lot of help from Claude, Anthropic's AI assistant (the name is a nod to that).
It is not an Anthropic model. Happy to answer anything about how it works or how we measured it.
