---
title: Debussy
emoji: 🌙
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Fast, honest decisions for AI agents, on a free CPU
models:
  - debussylabs/debussy
  - unsloth/Qwen3.5-4B-GGUF
---

# Debussy demo

Try [Debussy](https://huggingface.co/debussylabs/debussy), an open System 1 decision model for AI agents: safety
judgements, tool-call routing and knowledge questions with an honest "I don't know", each in a single pass.

This Space runs **Debussy Light** (Qwen3.5-4B, 4-bit) on Hugging Face's free 2-core CPU, so a decision takes several
seconds and requests wait in line. The recommended Debussy Duo scores higher; see the model card and
[debussylabs.com](https://debussylabs.com).
