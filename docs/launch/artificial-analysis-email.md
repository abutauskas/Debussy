**To:** Artificial Analysis (contact form or email on artificialanalysis.ai)
**Subject:** Independent evaluation request: Debussy, an open decision model (Decision Index)

Hello,

I'm the developer of Debussy, an open-source (Apache-2.0) decision model built for the format of your Decision Index:
one pass per decision and a probability for every option, with no free-form reasoning.

We have measured it ourselves on the four parts of the index and would very much value an independent evaluation.
Our estimates (methods and caveats are in the repository):
- When2Call: 93.2% (600 test questions)
- WildGuardTest: average F1 0.936 on 500 annotator-agreed rows
- HLE-MultiChoice: 15.0% (all 513 questions)
- AA-Omniscience: 53.1 on your public sample as a stand-in (the private set is unknown to us)
- Estimated index ≈ 63.7 for the recommended variant (Qwen3.5-4B for tool calls, K2-Horizon-7B for safety and knowledge)

Everything is public: code and heads at https://github.com/abutauskas/Debussy, model card at
https://huggingface.co/debussylabs/debussy. It runs locally with llama.cpp, and includes an HTTP server with one
endpoint per decision type (`python -m debussy.server`).

Could you tell me how decision models are submitted for evaluation, and what you would need from us (for example a
hosted endpoint, or a specific input/output format)? I'm happy to adapt.

Thank you,
[Your name]
Debussy Labs · https://debussylabs.com
