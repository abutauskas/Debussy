I'm launching Debussy, an open "System 1" decision model for AI agents.

Most of what an AI agent does all day is make small decisions: is this request safe, should it call this tool or ask
for the missing details first, does it actually know the answer? Debussy makes those calls in a single pass, with a
probability for every option, and says "I don't know" when it isn't sure.

It's built on two open models, Qwen3.5-4B and K2-Horizon-7B, with tiny calibrated heads on top, and it was developed
entirely on a 4-core CPU. Our own estimate puts it at about 63.7 on the Artificial Analysis Decision Index (not an
official score yet), and we've published what didn't work as carefully as what did.

It's open source (Apache-2.0): debussylabs.com

Built with a lot of help from Claude, Anthropic's AI assistant. Debussy is not an Anthropic model.
