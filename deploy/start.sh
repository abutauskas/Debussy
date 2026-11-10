#!/usr/bin/env bash
# Start Debussy Duo as an API on any machine: two llama-servers (K2-Horizon for safety and knowledge, Qwen3.5-4B for
# tool calls), then the Debussy HTTP API on $PORT. Uses the GPU if one is present (all layers offloaded).
#
#   DEBUSSY_GATEWAY_TOKEN=...  deploy/start.sh        (env: MODELS_DIR, PORT=8000, LLAMA_SERVER=llama-server, NGL)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO/models}"; PORT="${PORT:-8000}"; LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
if [ -z "${NGL:-}" ]; then if command -v nvidia-smi >/dev/null 2>&1; then NGL=999; else NGL=0; fi; fi
[ -n "${DEBUSSY_GATEWAY_TOKEN:-}" ] || echo "warning: DEBUSSY_GATEWAY_TOKEN is not set, so the API answers anyone who can reach it" >&2

python3 "$REPO/deploy/download_models.py" "$MODELS_DIR"
"$LLAMA_SERVER" -m "$MODELS_DIR/K2-Horizon-7B-Q4_K_M.gguf" -c 8192 --cache-ram 0 -np 1 -ngl "$NGL" --host 127.0.0.1 --port 8080 &
"$LLAMA_SERVER" -m "$MODELS_DIR/Qwen3.5-4B-Q4_K_M.gguf" -c 8192 --cache-ram 0 -np 1 -ngl "$NGL" --host 127.0.0.1 --port 8081 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
for port in 8080 8081; do
  for i in $(seq 1 300); do curl -s --max-time 2 "http://127.0.0.1:$port/health" | grep -q '"ok"' && break; sleep 2; done
  curl -s --max-time 2 "http://127.0.0.1:$port/health" | grep -q '"ok"' || { echo "llama-server on port $port did not start" >&2; exit 1; }
done
echo "models ready (GPU layers: $NGL); starting the Debussy API on port $PORT" >&2
cd "$REPO"
export DEBUSSY_K2_SERVER=http://127.0.0.1:8080 DEBUSSY_QWEN_SERVER=http://127.0.0.1:8081
exec python3 -m debussy.server --variant duo --host 0.0.0.0 --port "$PORT"
