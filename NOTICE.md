# Third-party notices

Debussy's own code is in this repository. It builds on, but does not include, the following:

| Component | Use | Licence / terms |
|---|---|---|
| Qwen3.5-4B (Qwen team, Alibaba) | current base model (GGUF via `unsloth/Qwen3.5-4B-GGUF`) | Apache-2.0 |
| K2-Horizon-7B (Institute of Foundation Models) | experimental base model (`IFM/K2-Horizon-7B-GGUF`) | Apache-2.0 |
| llama.cpp / llama-cpp-python | inference | MIT |
| Cormorant Garamond, Fraunces (Google Fonts) | wordmark lettering (converted to outlines in `brand/`) | SIL Open Font License 1.1 (`brand/FONT-LICENSE-OFL.txt`) |
| When2Call (NVIDIA) | training data + evaluation | see the dataset card on Hugging Face (`nvidia/When2Call`) |
| WildGuardMix (Ai2) | calibration data + evaluation | **gated**; accept the terms on Hugging Face; do not redistribute |
| Humanity's Last Exam (CAIS / Scale AI) | evaluation only | **gated**; evaluation only, never train on it, never publish its questions |
| MMLU-Pro (TIGER-Lab) | knowledge proxy for the "I don't know" threshold | see dataset card |
| AA-Omniscience-Public v1.1 (Artificial Analysis) | knowledge stand-in | Apache-2.0 |

Debussy was built by the project owner with help from Claude (Anthropic's AI assistant). It is not an Anthropic model.
