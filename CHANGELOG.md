# Releases

## 0.1.0 (November 6, 2026): first release

Debussy is an open **System 1 decision model** for AI agents. It makes three kinds of decision, each in a single pass
with a probability for every option: safety judgements, tool-call routing, and knowledge questions with an honest
"I don't know". Built and evaluated on a 4-core CPU with no GPU and no API key.

### Variants
| Variant | Base model(s) | RAM | Estimated Decision Index |
|---|---|---|---|
| `DebussyDuo()` (recommended) | Qwen3.5-4B for tool calls, K2-Horizon-7B for safety and knowledge | ~9 GB | **≈ 63.7** |
| `Debussy(profile="k2-horizon", backend="server")` | K2-Horizon-7B, thinking switched off | ~6 GB | ≈ 62.8 |
| `Debussy()` | Qwen3.5-4B | ~3 GB | ≈ 61.4 |

### Results (our estimates, not official Artificial Analysis scores)
| Benchmark | Debussy Duo | Debussy Light | Test set |
|---|---|---|---|
| When2Call (accuracy) | 93.2% | 93.2% | 600 test questions |
| WildGuardTest (average F1) | 0.936 | 0.905 | the same 500 annotator-agreed rows (Light: 0.899 on all 880) |
| AA-Omniscience stand-in | 53.1 | 51.8 | 266 public questions |
| HLE-MultiChoice (accuracy) | 15.0% | 10.5% | all 513 questions (chance 16.4%) |

### What's in it
- **Package:** `Debussy` and `DebussyDuo` (`debussy/model.py`), and an HTTP API server, `python -m debussy.server`
  (`/v1/wildguard`, `/v1/when2call`, `/v1/choice`).
- **Heads:**
  - `heads.json` (Qwen3.5-4B).
  - `heads_k2-horizon.json` (K2-Horizon-7B): safety heads fitted on 718 calibration rows, a tool-call head on 500
    training pairs, an "I don't know" threshold of 0.9, and a letter-bias correction.
- **Letter-bias correction:** K2-Horizon leans towards some answer letters (B) and away from the last option. A small
  correction, chosen on 600 practice questions where every letter is equally likely, took HLE-MultiChoice from
  13.6% to 15.0%.
- **debussylabs.com:** model page, news, API docs, privacy notice. Sign-in with GitHub, self-serve API keys and the `/v1`
  gateway are built but switched off until the hosted API opens (`SIGNUPS_OPEN`).

### Checked
- **Package matches the research pipeline.** Qwen: 3/3 tool-call and 9/9 safety decisions. K2-Horizon: 5/5 and 15/15,
  with a largest probability difference of 0.0000.
- **The rules:** every setting was chosen on training or practice data only, and each test set was used once, to
  report. For the last two experiments the adoption rule (nested cross-validation) was fixed before seeing any test
  data.

### What didn't work
- A probe on K2-Horizon's internal state for knowledge questions lost to its plain confidence (51.8 vs 53.1).
- A probe on K2-Horizon's internal state for the safety judge beat the heads inside the calibration data but not on
  the test rows (average F1 0.927 vs 0.936), and it would double the time per judgement.
- Combining the two models' knowledge answers, and Wikipedia retrieval for obscure questions, gave little or nothing.
- Both variants are below chance on HLE-MultiChoice.

### Known limitations
- About 13 s per safety judgement on a 4-core CPU; GPU and laptop speeds are not measured yet.
- English only; the safety judge reads the first 600 tokens of the request and of the reply.
- WildGuardTest scores use the annotator-agreed rows (as Artificial Analysis does), so they are not comparable with
  guard-model results on the full test set.
- One signal for a safety system, not a complete one.

Built by Debussy Labs with help from Claude, Anthropic's AI assistant. Debussy is not an Anthropic model.
Base models: Qwen3.5-4B (Qwen team) and K2-Horizon-7B (Institute of Foundation Models), both Apache-2.0.
