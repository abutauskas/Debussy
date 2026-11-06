"""Publish Debussy to Hugging Face: the model card, both heads files, the logo, licence, notices and changelog.

usage: python scripts/release/hf_upload.py [--repo debussylabs/debussy] [--dry-run]
       python scripts/release/hf_upload.py --org-card [--dry-run]   (the organisation page: Space debussylabs/README)
       python scripts/release/hf_upload.py --demo [--dry-run]       (the demo: Space debussylabs/debussy-demo)
Needs a Hugging Face token with write access to the organisation, in HF_TOKEN (in a Claude cloud session the network
proxy injects the saved token, so HF_TOKEN can be any placeholder). Creates the repo (public) if it doesn't exist and
uploads everything in one commit named after the package version. No model weights are uploaded: the card links to
the base models on Hugging Face.
"""
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
FILES = {  # path in the Hugging Face repo -> path here
    "README.md": "hf/README.md",
    "heads.json": "debussy/heads.json",
    "heads_k2-horizon.json": "debussy/heads_k2-horizon.json",
    "debussy-logo.svg": "brand/debussy-logo.svg",
    "LICENSE": "LICENSE",
    "NOTICE.md": "NOTICE.md",
    "CHANGELOG.md": "CHANGELOG.md",
}

ORG_CARD = {  # Hugging Face shows an organisation's README from its Space named "README"
    "README.md": "hf/org/README.md", "index.html": "hf/org/index.html", "debussy-avatar.png": "brand/debussy-avatar-square-512.png",
    "debussy-labs-banner.png": "brand/debussy-labs-banner.png",
}

DEMO = {  # the Gradio demo ships with the released Debussy code and heads; the base model is downloaded at start-up
    "README.md": "hf/space/README.md", "app.py": "hf/space/app.py", "requirements.txt": "hf/space/requirements.txt",
    **{f"debussy/{f}": f"debussy/{f}" for f in ["__init__.py", "model.py", "engine.py", "paths.py", "heads.json", "heads_k2-horizon.json"]},
    **{f"debussy/skills/{f}": f"debussy/skills/{f}" for f in ["__init__.py", "when2call.py", "wildguard.py"]},
}

if __name__ == "__main__":
    org = "--org-card" in sys.argv; demo = "--demo" in sys.argv
    if org: FILES = ORG_CARD
    if demo: FILES = DEMO
    default = "debussylabs/debussy-demo" if demo else "debussylabs/README" if org else "debussylabs/debussy"
    repo = sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else default   # e.g. --demo --repo K00lKidd/debussy-demo
    from debussy import __version__
    for dst, src in FILES.items():
        print(f"  {src:32s} -> {repo}/{dst}  ({os.path.getsize(os.path.join(ROOT, src)):,} bytes)")
    if "--dry-run" in sys.argv: sys.exit(0)
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or "proxy-injected")
    kind = "space" if (org or demo) else "model"
    sdk = {"space_sdk": "static"} if org else {"space_sdk": "gradio"} if demo else {}
    url = api.create_repo(repo, repo_type=kind, private=False, exist_ok=True, **sdk)
    info = api.create_commit(repo, repo_type=kind, commit_message=("Organisation card" if org else f"Debussy {__version__} demo" if demo else f"Debussy {__version__}"),
                             commit_description=("README, landing page and logo." if org else "Gradio app with the released Debussy package." if demo else "Model card, calibrated heads, logo, licence, notices and changelog."),
                             operations=[CommitOperationAdd(dst, os.path.join(ROOT, src)) for dst, src in FILES.items()])
    print("repo:", url, "\ncommit:", info.commit_url)
