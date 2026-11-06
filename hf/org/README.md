---
title: README
emoji: 🌙
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
---

<p align="center"><img src="https://huggingface.co/spaces/debussylabs/README/resolve/main/debussy-avatar.png" width="88" alt="Debussy Labs logo"></p>

**Debussy Labs** builds fast, honest decision models for AI agents.

Our first model, **Debussy**, is an open *System 1* decision model. It makes the quick calls an agent needs (is this
request safe, should I call this tool, do I actually know this) in a single pass on an ordinary CPU, and it says
"I don't know" when it doesn't.

| Model | What it decides | Licence |
|---|---|---|
| [**debussy**](https://huggingface.co/debussylabs/debussy) | Safety judgements · tool-call routing · knowledge questions with "I don't know" | Apache-2.0 |

**Estimated Decision Index ≈ 63.7** on the Artificial Analysis Decision Index (our own estimate, not an official
score). Built and evaluated on a 4-core CPU, with no GPU and no API key.

🌐 [debussylabs.com](https://debussylabs.com) · 💻 [GitHub](https://github.com/abutauskas/Debussy) ·
📰 [Introducing Debussy](https://debussylabs.com/news/introducing-debussy/) · 📦 [Releases](https://debussylabs.com/releases/)

<sub>Built with help from Claude, Anthropic's AI assistant. Debussy is not an Anthropic model.</sub>
