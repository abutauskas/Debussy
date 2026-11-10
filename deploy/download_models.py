"""Download Debussy Duo's two base models (GGUF) into MODELS_DIR, once. Used by start.sh and the Modal app.

usage: python deploy/download_models.py [MODELS_DIR]      (default: models/ next to this repository)
"""
import os, sys
from huggingface_hub import hf_hub_download

MODELS = [("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"), ("IFM/K2-Horizon-7B-GGUF", "K2-Horizon-7B-Q4_K_M.gguf")]


def ensure(models_dir):
    os.makedirs(models_dir, exist_ok=True)
    for repo, name in MODELS:
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            print(f"downloading {repo}/{name} ...", flush=True)
            hf_hub_download(repo, name, local_dir=models_dir)
        print(f"ok {path} ({os.path.getsize(path) / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    ensure(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
