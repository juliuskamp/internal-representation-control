"""Neuronpedia auto-interp label lookups for the pinned SAE variant.

Neuronpedia indexed gemma-3-27b-it under source "{layer}-gemmascope-2-res-16k",
which is exactly the pinned SAE release (see irc/constants.py) — latent indices
line up with its labels only for that variant.
"""

import json
import os
import urllib.request

NEURONPEDIA_MODEL = "gemma-3-27b-it"
NEURONPEDIA_SOURCE_TEMPLATE = "{layer}-gemmascope-2-res-16k"


def fetch_label(layer: int, index: int, timeout: float = 20.0) -> str:
    """Auto-interp explanation for one latent. Raises on network failure
    (caller decides whether that is fatal or just uncacheable)."""
    url = (f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/"
           f"{NEURONPEDIA_SOURCE_TEMPLATE.format(layer=layer)}/{index}")
    headers = {"User-Agent": "irc-pipeline"}
    if os.environ.get("NEURONPEDIA_API_KEY"):
        headers["x-api-key"] = os.environ["NEURONPEDIA_API_KEY"]
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    exps = data.get("explanations") or []
    return exps[0].get("description", "") if exps else "(no explanation)"
