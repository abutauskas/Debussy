"""Debussy Duo as a serverless GPU API on Modal (modal.com): you pay only while requests run.

One-time setup, from any terminal with Python (your PC or a GitHub Codespace on this repository):
    pip install modal
    modal token new                                                   # log in (opens a browser link)
    python3 -c "import secrets; print(secrets.token_urlsafe(32))"     # make a gateway token; keep it
    modal secret create debussy DEBUSSY_GATEWAY_TOKEN=<that token>
    modal deploy deploy/modal_app.py                                  # prints the API's web address

Then set INFERENCE_URL (that address) and INFERENCE_TOKEN (the same token) in Cloudflare Pages; see docs/DEPLOY.md.

How it runs: llama.cpp's official CUDA server image with both models on one GPU (deploy/start.sh). The models are
downloaded once into a Modal Volume. The GPU stops after 5 idle minutes (the next request then takes a few minutes
to start), and at most one GPU runs at a time, which caps the cost.
"""
import os
import modal

GPU = os.environ.get("DEBUSSY_GPU", "T4")        # 16 GB fits both 4-bit models; set DEBUSSY_GPU=L4 for more speed
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
    .entrypoint([])                                # the image normally starts llama-server; start.sh does that
    .apt_install("curl")
    .pip_install("numpy", "pandas", "huggingface_hub")
    .add_local_dir(os.path.join(REPO, "debussy"), "/root/app/debussy", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(os.path.join(REPO, "deploy"), "/root/app/deploy", ignore=["__pycache__", "*.pyc"])
)
models = modal.Volume.from_name("debussy-models", create_if_missing=True)
app = modal.App("debussy")


@app.function(image=image, gpu=GPU, volumes={"/models": models}, secrets=[modal.Secret.from_name("debussy")],
              scaledown_window=300, timeout=24 * 3600, max_containers=1)
@modal.concurrent(max_inputs=8)                  # requests queue inside; the API answers one decision at a time
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    import subprocess
    env = dict(os.environ, MODELS_DIR="/models", LLAMA_SERVER="/app/llama-server", PORT="8000", NGL="999")
    subprocess.Popen(["bash", "/root/app/deploy/start.sh"], env=env)
