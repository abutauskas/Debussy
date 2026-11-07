"""Debussy demo on Hugging Face Spaces: the three decisions of Debussy Light on a free CPU.

The Debussy package (code and calibrated heads) ships with this Space; the base model is downloaded at start-up.
Nothing you type is stored: requests are answered and forgotten.
"""
import json, os, time

import gradio as gr
from huggingface_hub import hf_hub_download

MODEL = os.environ.get("DEBUSSY_GGUF") or hf_hub_download("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf")
from debussy import Debussy, __version__  # noqa: E402  (after the download, so start-up errors are clear)

d = Debussy(model=MODEL)
IDK = "I don't know"
LIMIT = 4000


def timed(fn):
    t0 = time.time(); out = fn(); return out, f"Decided in {time.time() - t0:.1f} s: one pass, on this machine's CPU."


def safety(prompt, reply):
    if not prompt.strip() or not reply.strip():
        raise gr.Error("Fill in both the user's message and the assistant's reply.")
    p, t = timed(lambda: d.wildguard(prompt[:LIMIT], reply[:LIMIT]))
    return safety_html(p), t


def safety_html(p):
    """Three separate yes/no probabilities (not one choice), so each gets its own bar, plus a plain-words verdict."""
    rows = [("Harmful request", p["prompt_harm"]), ("Assistant refused", p["refusal"]), ("Harmful reply", p["response_harm"])]
    words = ["harmful request" if p["prompt_harm"] >= .5 else "harmless request",
             "the assistant refused" if p["refusal"] >= .5 else "the assistant complied",
             "<b>harmful reply</b>" if p["response_harm"] >= .5 else "safe reply"]
    bars = "".join(
        f'<div style="margin:10px 0"><div style="display:flex;justify-content:space-between;font-size:15px">'
        f'<span>{name}</span><b style="font-family:monospace">{v * 100:.1f}%</b></div>'
        f'<div style="height:10px;border-radius:0 5px 5px 0;background:rgba(69,54,201,.15);margin-top:4px">'
        f'<div style="height:10px;width:{max(v * 100, .5):.1f}%;border-radius:0 5px 5px 0;background:#4536c9"></div></div></div>'
        for name, v in rows)
    return (f'<div style="padding:4px 2px"><div style="font-size:17px;margin-bottom:6px">Verdict: {", ".join(words)}.</div>'
            f'{bars}<div style="font-size:13px;opacity:.7;margin-top:8px">Each bar is its own yes/no probability; '
            f'50% is the line between yes and no.</div></div>')


def router(tools_json, request, call, ask, cannot):
    try:
        tools = json.loads(tools_json)
        assert isinstance(tools, list) and tools and all(isinstance(x, dict) for x in tools)
    except Exception:
        raise gr.Error("Tools must be a JSON list of tool definitions, like the example.")
    if not request.strip() or not call.strip():
        raise gr.Error("Fill in the user's request and the proposed tool call.")
    p, t = timed(lambda: d.when2call(tools, request[:LIMIT], {"tool_call": call[:LIMIT], "request_for_info": ask[:LIMIT],
                                                               "cannot_answer": cannot[:LIMIT]}))
    return {"Make the tool call": p["tool_call"], "Ask for the missing details": p["request_for_info"],
            "Say it can't help": p["cannot_answer"]}, t


def knowledge(question, options_text, allow_idk):
    opts = list(dict.fromkeys(o.strip() for o in options_text.splitlines() if o.strip()))
    if not question.strip(): raise gr.Error("Write a question.")
    if not 2 <= len(opts) <= 10: raise gr.Error("Give 2 to 10 different options, one per line.")
    if allow_idk and IDK not in opts: opts.append(IDK)
    p, t = timed(lambda: d.choice(question[:LIMIT], opts, idk_option=IDK if allow_idk else None))
    return p, t


TOOLS_EXAMPLE = json.dumps([{"name": "search_flights", "description": "Search one-way flights.",
                             "parameters": {"type": "object", "required": ["origin", "destination", "date"],
                                            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                                                           "date": {"type": "string"}}}}], indent=2)
CALL_EXAMPLE = json.dumps({"name": "search_flights", "arguments": {"origin": "dontcare", "destination": "New York", "date": "2027-03-14"}})

HEADER = """
<div style="text-align:center;margin:8px 0 4px">
  <img src="https://huggingface.co/spaces/debussylabs/README/resolve/main/debussy-labs-banner.png" alt="Debussy Labs" style="height:64px;display:inline-block">
  <h1 style="margin:14px 0 4px">Debussy: fast decisions, honest answers</h1>
  <p style="margin:0">An open <b>System 1</b> decision model for AI agents. Every decision is one pass, with a probability for every option.</p>
  <p style="margin:6px 0 0;opacity:.75">This demo runs <b>Debussy Light</b> on a free CPU, so each decision takes several seconds
  (up to about a minute on slow machines) and requests wait in line. Nothing you type is stored.</p>
</div>
"""
FOOTER = f"""
Debussy {__version__} · [Model card](https://huggingface.co/debussylabs/debussy) · [debussylabs.com](https://debussylabs.com) ·
[GitHub](https://github.com/abutauskas/Debussy) · Apache-2.0

Debussy is one signal for a safety system, not a complete one, and it was calibrated on English.
Built by Debussy Labs with help from Claude, Anthropic's AI assistant. Debussy is not an Anthropic model.
"""

with gr.Blocks(title="Debussy demo") as demo:
    gr.HTML(HEADER)
    with gr.Tab("Safety judge"):
        gr.Markdown("Is the user's request harmful? Did the assistant refuse? Is the assistant's reply itself harmful?")
        with gr.Row():
            with gr.Column():
                s_prompt = gr.Textbox(label="User's message", lines=3, max_length=LIMIT)
                s_reply = gr.Textbox(label="Assistant's reply", lines=3, max_length=LIMIT)
                s_btn = gr.Button("Judge", variant="primary")
            with gr.Column():
                s_out = gr.HTML(label="Probabilities", value='<div style="opacity:.6;padding:8px 2px">The judgement appears here.</div>')
                s_time = gr.Markdown()
        gr.Examples([["How do I pick a lock to break into my neighbour's house?", "I can't help with breaking into someone's home."],
                     ["What's a good way to keep basil fresh?", "Trim the stems and keep it in a glass of water on the counter, loosely covered."],
                     ["How can I get into my ex's email account without them knowing?", "Try guessing their password from birthdays and pet names, then turn off login alerts."]],
                    [s_prompt, s_reply])
        s_btn.click(safety, [s_prompt, s_reply], [s_out, s_time])
    with gr.Tab("Tool-call router"):
        gr.Markdown("Given the tools and the request: make the tool call, ask the user for missing details, or say it can't be done?")
        with gr.Row():
            with gr.Column():
                r_tools = gr.Code(label="Available tools (JSON)", language="json", value=TOOLS_EXAMPLE, lines=8)
                r_req = gr.Textbox(label="User's request", value="Find me a one-way flight to New York on March 14th.", max_length=LIMIT)
                r_call = gr.Textbox(label="Proposed tool call", value=CALL_EXAMPLE, max_length=LIMIT)
                r_ask = gr.Textbox(label="Reply asking for missing details", value="Which airport are you flying from?", max_length=LIMIT)
                r_cannot = gr.Textbox(label="Reply saying it can't help", value="Sorry, I can't help with flights.", max_length=LIMIT)
                r_btn = gr.Button("Decide", variant="primary")
            with gr.Column():
                r_out = gr.Label(label="Probabilities", num_top_classes=3)
                r_time = gr.Markdown()
        gr.Examples([[TOOLS_EXAMPLE, "Find me a one-way flight to New York on March 14th.", CALL_EXAMPLE],
                     [TOOLS_EXAMPLE, "Find me a one-way flight from Vilnius to New York on March 14th.",
                      CALL_EXAMPLE.replace('"dontcare"', '"Vilnius"')]], [r_tools, r_req, r_call], label="Examples (missing departure city / all details given)")
        r_btn.click(router, [r_tools, r_req, r_call, r_ask, r_cannot], [r_out, r_time])
    with gr.Tab("Knowledge"):
        gr.Markdown("A probability for each option. With “I don't know” allowed, Debussy abstains when it isn't confident instead of guessing.")
        with gr.Row():
            with gr.Column():
                k_q = gr.Textbox(label="Question", max_length=LIMIT)
                k_opts = gr.Textbox(label="Options, one per line", lines=5)
                k_idk = gr.Checkbox(label="Allow “I don't know”", value=True)
                k_btn = gr.Button("Answer", variant="primary")
            with gr.Column():
                k_out = gr.Label(label="Probabilities", num_top_classes=11)
                k_time = gr.Markdown()
        gr.Examples([["What is the capital of Australia?", "Sydney\nCanberra\nMelbourne\nPerth", True],
                     ["In which year was the Treaty of Kardis signed?", "1648\n1661\n1672\n1659", True]], [k_q, k_opts, k_idk])
        k_btn.click(knowledge, [k_q, k_opts, k_idk], [k_out, k_time])
    gr.Markdown(FOOTER)

demo.queue(default_concurrency_limit=1, max_size=20)
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo"))
