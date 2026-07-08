"""Environment bootstrap. Import this before any huggingface/transformers import.

Loads .env (HF_TOKEN, HF_HOME). huggingface_hub reads HF_HOME at import time,
so this module must run first in every entry point.
"""

import os

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HOME", "/workspace/hf-cache")

if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("HF_TOKEN is not set — add it to .env at the repo root")
