"""Where Debussy finds things. Everything is relative to the repository root, so it runs from any folder."""
import os

PACKAGE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE)
DATA = os.path.join(ROOT, "data")        # datasets + per-question results (not in git; see data/README.md)
MODELS = os.path.join(ROOT, "models")    # GGUF model files (not in git)
HEADS = os.path.join(PACKAGE, "heads.json")
