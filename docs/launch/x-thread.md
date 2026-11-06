1/ Meet Debussy 🌙 an open "System 1" decision model for AI agents. Is this safe? Should I call this tool? Do I
actually know this? One pass per decision, a probability for every option, on an ordinary CPU. debussylabs.com

2/ No chain of thought, no generation. A small open model reads the input once and answers short yes/no questions;
tiny calibrated heads turn that into decisions. Base models unchanged. Built on a 4-core CPU, no GPU.

3/ Surprise: two small models beat one. Qwen3.5-4B is the better tool-caller, K2-Horizon-7B (thinking off) the
sharper safety judge. "Debussy Duo" routes each decision to its specialist.

4/ Our estimates (not official): tool calls 93.2%, safety F1 0.936, honest knowledge 53.1, and HLE 15.0%, which is
below chance. A System 1 model doesn't solve HLE, and we say so.

5/ We published what didn't work too: two internal-state probes, model ensembling, retrieval. One probe looked great
in calibration and lost on test. Honesty > hype.

6/ Apache-2.0. Code: github.com/abutauskas/Debussy · Model: huggingface.co/debussylabs/debussy
Built with help from Claude, not an Anthropic model.
