# Running the Debussy API server

The website, sign-in, keys and gateway run on Cloudflare (`docs/HOSTING.md`). The model itself runs on a machine
you control, with `python -m debussy.server`. This page covers where that machine can be and how it connects.

| Where | Cost | Speed per decision | Good for |
|---|---|---|---|
| **Your own PC** | free (electricity) | CPU: roughly 1–13 s; NVIDIA GPU: probably well under 1 s (not yet measured) | a beta, the demo |
| **A GPU server** (hourly rental) | paid per hour | probably well under 1 s (not yet measured) | the paid API |
| **Modal** (serverless GPU, `deploy/modal_app.py`) | pay per second while busy; $30 free usage a month | probably well under 1 s once warm | opening the API |
| A free CPU server (e.g. Oracle Always Free) | free | about 1–13 s | a beta |

Whichever you choose, the steps are the same: **(1) run the server, (2) connect it with a Cloudflare Tunnel,
(3) tell the website where it is.** The server never needs an open port.

## The quickest way: Modal

`deploy/modal_app.py` runs Debussy Duo on a Modal GPU with its own web address, so no tunnel is needed. From any
terminal with Python (your PC, or a GitHub Codespace on this repository):

```bash
pip install modal
modal token new                                                   # log in to Modal
python3 -c "import secrets; print(secrets.token_urlsafe(32))"     # a gateway token: copy it, keep it private
modal secret create debussy DEBUSSY_GATEWAY_TOKEN=<that token>
modal deploy deploy/modal_app.py                                  # prints https://<you>--debussy-serve.modal.run
```

- The first request after a deploy downloads both models into a Modal Volume and starts the GPU (several minutes);
  after 5 idle minutes the GPU stops, and the next request takes a minute or two to start it again.
- At most one GPU runs at a time (`max_containers=1`), which caps the cost. Set a spending limit in Modal's settings too.
- Check it: `curl https://<you>--debussy-serve.modal.run/health`.
- Then do step 3 below with that address as `INFERENCE_URL` and the same token as `INFERENCE_TOKEN`.

The machine-independent version of the same thing is `deploy/start.sh` (both models + the API, GPU if present).

## 1. Run the server

Needs Python 3.10+ and, for the Duo, about 9 GB of free memory (Light: about 3 GB).

```bash
git clone https://github.com/abutauskas/Debussy && cd Debussy
pip install llama-cpp-python==0.3.36 numpy pandas pyarrow scikit-learn \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
mkdir -p models && curl -L -o models/Qwen3.5-4B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
export DEBUSSY_GATEWAY_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"   # keep this value
python -m debussy.server --variant light --host 127.0.0.1 --port 8000
```

- **Windows:** the same commands work in PowerShell (use `set` / `$env:` for the variable); the llama.cpp package has
  a ready-made Windows build.
- **The Duo** also needs K2-Horizon served by llama.cpp's `llama-server` (download a release build for your system from
  github.com/ggml-org/llama.cpp/releases, CUDA build for NVIDIA cards), started with
  `llama-server -m models/K2-Horizon-7B-Q4_K_M.gguf -c 10240 --cache-ram 0 --port 8080`, and the model file from
  `IFM/K2-Horizon-7B-GGUF` on Hugging Face. Then run `python -m debussy.server --variant duo`.
- **NVIDIA GPU:** install the CUDA build of llama-cpp-python instead of the CPU one (see the llama-cpp-python README)
  and the CUDA build of `llama-server`. *Not yet tested by us: the first GPU run should be measured before pricing.*

Check it: `curl http://127.0.0.1:8000/health` should answer `{"status": "ok", ...}`.

## 2. Connect it with a Cloudflare Tunnel

1. Cloudflare dashboard → **Zero Trust → Networks → Tunnels → Create a tunnel** (type *Cloudflared*), name it
   `debussy-model`.
2. Choose your system; Cloudflare shows an install command with a token. Run it on the machine. (Treat that token like
   a password.)
3. Add a **public hostname**: `model.debussylabs.com` → service `http://127.0.0.1:8000`.
4. Check from anywhere: `https://model.debussylabs.com/health`.

## 3. Tell the website where it is

Cloudflare → Workers & Pages → your project → **Settings → Variables and secrets**:
- `INFERENCE_URL` = `https://model.debussylabs.com` (plain text)
- `INFERENCE_TOKEN` = the `DEBUSSY_GATEWAY_TOKEN` value from step 1 (type **Secret**)

Redeploy. API calls with a valid key now reach the model; everything else (sign-ups, keys, limits) works as described in
`docs/HOSTING.md`. Sign-ups stay closed until you set `SIGNUPS_OPEN=true`.

## Keeping it running
- Run the server and the tunnel as services that restart after a reboot (systemd on Linux, a scheduled task or NSSM on
  Windows, launchd on macOS).
- The server handles one decision at a time. If many people use it, run more copies behind the tunnel, or a GPU.
- Logs record only the time, endpoint and status code, never request contents (as the privacy notice promises).
