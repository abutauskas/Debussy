"""Load KEY=VALUE pairs from decision-model/.env into the environment (real env vars win)."""
import os
from pathlib import Path

def load_env(path=Path(__file__).with_name(".env")):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip() and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

load_env()
