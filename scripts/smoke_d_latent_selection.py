"""Smoke test D: data-driven SAE latent selection for "football" at layer 31,
cross-checked against Neuronpedia auto-interp labels.

Selection: rank latents by (mean max-activation on concept texts) minus
(mean max-activation on baseline texts), requiring near-zero baseline activity.
"""

from irc import env  # noqa: F401

import json
import os
import urllib.request

import torch
from sae_lens import SAE

from irc.constants import SAE_ID_TEMPLATE, SAE_RELEASE
from irc.model import ResidualCapture, chat_ids, load_model, load_tokenizer

# Pinned SAE variant: latent indices line up with Neuronpedia's auto-interp
# labels ("{layer}-gemmascope-2-res-16k"); see irc/constants.py.
RELEASE = SAE_RELEASE
LAYER = 31
SAE_ID = SAE_ID_TEMPLATE.format(layer=LAYER)
TOP_K = 8

CONCEPT_TEXTS = [
    "Tell me about football.",
    "The striker scored a goal in the final minute of the football match.",
    "He practices football with his team every Saturday afternoon.",
    "The quarterback threw a long pass down the football field.",
]
BASELINE_TEXTS = [
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Interest rates were left unchanged by the central bank on Thursday.",
    "She painted the old fence a bright shade of blue last weekend.",
    "The museum's new exhibit features pottery from the twelfth century.",
    "Clouds gathered over the valley as the hikers reached the summit.",
    "The committee approved the budget after a lengthy discussion.",
]


@torch.no_grad()
def max_feature_acts(model, tokenizer, sae, texts: list[str]) -> torch.Tensor:
    """(n_texts, d_sae): per-text max SAE activation over content tokens."""
    rows = []
    for text in texts:
        ids = chat_ids(tokenizer, text)
        with ResidualCapture(model, [LAYER]) as cap:
            model(ids)
        resid = cap.acts[LAYER][0].to("cuda", sae.dtype)  # (seq, d_model)
        feats = sae.encode(resid)  # (seq, d_sae)
        rows.append(feats.max(dim=0).values.float().cpu())
    return torch.stack(rows)


def neuronpedia_label(layer: int, index: int) -> str:
    """Fetch the auto-interp explanation for one latent; message string on failure."""
    url = f"https://www.neuronpedia.org/api/feature/gemma-3-27b-it/{layer}-gemmascope-2-res-16k/{index}"
    headers = {"User-Agent": "smoke-test"}
    if os.environ.get("NEURONPEDIA_API_KEY"):
        headers["x-api-key"] = os.environ["NEURONPEDIA_API_KEY"]
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
    except Exception as e:
        return f"(neuronpedia lookup failed: {e})"
    exps = data.get("explanations") or []
    if exps:
        return exps[0].get("description", "")
    return "(no explanation on record)"


def main() -> None:
    result = SAE.from_pretrained(RELEASE, SAE_ID, device="cuda")
    sae = result[0] if isinstance(result, tuple) else result
    tokenizer = load_tokenizer()
    model = load_model()

    concept_acts = max_feature_acts(model, tokenizer, sae, CONCEPT_TEXTS)
    baseline_acts = max_feature_acts(model, tokenizer, sae, BASELINE_TEXTS)

    concept_mean = concept_acts.mean(dim=0)
    baseline_mean = baseline_acts.mean(dim=0)
    specificity = concept_mean - baseline_mean

    top = torch.topk(specificity, TOP_K)
    print(f"top {TOP_K} latents by specificity (concept_mean - baseline_mean):\n")
    print(f"{'latent':>7} {'concept':>8} {'baseline':>9} {'spec':>7}  label")
    results = []
    for idx, spec in zip(top.indices.tolist(), top.values.tolist()):
        label = neuronpedia_label(LAYER, idx)
        mentions = any(k in label.lower() for k in ("football", "soccer", "sport"))
        flag = " <-- label matches" if mentions else ""
        print(f"{idx:>7} {concept_mean[idx]:8.2f} {baseline_mean[idx]:9.2f} "
              f"{spec:7.2f}  {label}{flag}")
        results.append({"latent": idx, "concept_mean": float(concept_mean[idx]),
                        "baseline_mean": float(baseline_mean[idx]),
                        "specificity": spec, "label": label, "label_matches": mentions})

    n_match = sum(r["label_matches"] for r in results)
    print(f"\n{n_match}/{TOP_K} top latents have concept-related Neuronpedia labels")


if __name__ == "__main__":
    main()
