"""Smoke test B: load one Gemma Scope 2 SAE (mid-depth resid_post) via SAELens,
run sentence activations through it, confirm shapes and hook names line up.
"""

from irc import env  # noqa: F401

import torch
from sae_lens import SAE

from irc.constants import SAE_ID_TEMPLATE, SAE_RELEASE
from irc.model import ResidualCapture, chat_ids, load_model, load_tokenizer

# Pinned SAE variant (matches Neuronpedia's index; see irc/constants.py).
RELEASE = SAE_RELEASE
LAYER = 31
SAE_ID = SAE_ID_TEMPLATE.format(layer=LAYER)


def main() -> None:
    result = SAE.from_pretrained(RELEASE, SAE_ID, device="cuda")
    sae = result[0] if isinstance(result, tuple) else result
    print(f"SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, dtype={sae.cfg.dtype}")
    meta = getattr(sae.cfg, "metadata", None)
    if meta is not None:
        print(f"hook_name={getattr(meta, 'hook_name', '?')}, model={getattr(meta, 'model_name', '?')}")

    tokenizer = load_tokenizer()
    model = load_model()

    ids = chat_ids(tokenizer, "Tell me about football.")
    with ResidualCapture(model, [LAYER]) as cap, torch.no_grad():
        model(ids)
    resid = cap.acts[LAYER]  # (1, seq, d_model) fp32 cpu
    print(f"resid shape: {tuple(resid.shape)} (d_model should equal d_in={sae.cfg.d_in})")

    feats = sae.encode(resid.to("cuda", sae.dtype))
    print(f"SAE feature shape: {tuple(feats.shape)}")
    last = feats[0, -1]
    active = int((last > 0).sum())
    top = torch.topk(last, 10)
    print(f"active latents at last token: {active} (L0)")
    print("top-10 latent ids:", top.indices.tolist())
    print("top-10 activations:", [round(v, 2) for v in top.values.tolist()])

    # BOS (position 0) is a huge outlier the SAE was not trained on — exclude it,
    # as all downstream analyses must.
    recon = sae.decode(feats)
    err = (recon - resid.to("cuda", sae.dtype)).norm(dim=-1) / resid.cuda().norm(dim=-1)
    print(f"relative reconstruction error (excl. BOS): {err[0, 1:].mean().item():.3f}")


if __name__ == "__main__":
    main()
