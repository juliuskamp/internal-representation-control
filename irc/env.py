"""Environment bootstrap. Import this before any huggingface/transformers import.

Loads .env (HF_TOKEN, HF_HOME). huggingface_hub reads HF_HOME at import time,
so this module must run first in every entry point.
"""

import os

from dotenv import load_dotenv

from irc.paths import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

os.environ.setdefault("HF_HOME", "/workspace/hf-cache")


def require_hf_token() -> None:
    """Fail fast with a clear message before hitting a gated HF repo.

    Called at model/tokenizer/SAE load time (not import time), so entry
    points that never touch Hugging Face — e.g. nla_judge — run without it.
    """
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set — add it to .env at the repo root")
