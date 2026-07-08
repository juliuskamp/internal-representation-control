"""Export per-token representation-strength data and build the interactive viewer.

For every stored (word, sentence, condition) run of a pipeline run:
  - concept-vector cosines per layer x token for the target word (both variants)
  - control-word null band (mean ± std across the 100 control words) per layer x
    token, computed per condition (paper-style: one band per condition line)
  - SAE activation per layer x selected latent x token (aggregation and latent
    toggling happen client-side in the viewer)
All series are trimmed to the sentence's own tokens (the model sometimes emits a
trailing whitespace token before <end_of_turn>; verified to be the only length
mismatch, and responses always begin directly with the sentence tokens).

Writes the data (gzip+base64) into viz/repr_viewer.html's __DATA_B64__ slot and
saves the self-contained page to artifacts/runs/{run_id}/results/repr_viewer.html.

Usage: uv run python scripts/export_viz_data.py --run-id run1-core
"""

from irc import env  # noqa: F401

import base64
import dataclasses
import gzip
import json
from pathlib import Path

import torch
import tyro

from irc.pipeline import _load_records, _load_saes
from irc.words_paper import CONTROL_WORDS_PAPER

VARIANTS = ("paper", "word_tokens")
SAE_LAYERS = [16, 31, 40, 53]


@dataclasses.dataclass
class Config:
    run_id: str = "run1-core"


def rnd(t: torch.Tensor, nd: int = 3) -> list:
    return [[round(float(x), nd) for x in row] for row in t]


@torch.no_grad()
def main(cfg: Config) -> None:
    run_dir = Path("artifacts/runs") / cfg.run_id
    records = _load_records(run_dir)

    from transformers import AutoTokenizer

    from irc.model import MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    banks = {}
    for v in VARIANTS:
        bank = torch.load(Path("artifacts/concept_vectors") / f"bank_{v}_v1.pt")
        words = list(bank["vectors"].keys())
        V = torch.stack([bank["vectors"][w] for w in words]).cuda()
        banks[v] = {"words": words, "Vn": V / V.norm(dim=-1, keepdim=True)}
    ctrl_set = set(CONTROL_WORDS_PAPER)

    saes = _load_saes(SAE_LAYERS)
    latents_dir = Path("artifacts/latents_v1")

    data: dict = {}
    for i, rec in enumerate(records):
        si = rec["sentence_idx"]
        # word-free conditions (no_mention) apply to every word sharing the sentence
        target_words = (
            [rec["word"]] if rec["word"] else
            sorted({r["word"] for r in records if r["word"] and r["sentence_idx"] == si})
        )
        for word in target_words:
            slot = data.setdefault(word, {}).setdefault(str(si), {
                "sentence": rec["sentence"],
                "tokens": [
                    t.replace("▁", " ")
                    for t in tokenizer.convert_ids_to_tokens(
                        tokenizer(rec["sentence"], add_special_tokens=False)["input_ids"])
                ],
                "conditions": {},
            })
            cond_entry: dict = {"exact": rec["exact_match"], "completion": rec["completion"]}
            if rec["acts_file"]:
                A = torch.load(run_dir / rec["acts_file"]).float().cuda()  # (L,T,D)
                n_tok = len(slot["tokens"])
                assert A.shape[1] >= n_tok, f"{rec['key']}: acts shorter than sentence"
                A = A[:, :n_tok]  # drop trailing whitespace token(s)
                An = A / A.norm(dim=-1, keepdim=True)
                for v in VARIANTS:
                    b = banks[v]
                    cos = torch.einsum("ltd,wld->lwt", An, b["Vn"])  # (L,W,T)
                    tgt = b["words"].index(word)
                    ctrl = torch.tensor(
                        [j for j, w in enumerate(b["words"]) if w in ctrl_set])
                    cc = cos[:, ctrl]  # (L, n_control, T)
                    null_mean = cc.mean(dim=1)
                    null_std = cc.std(dim=1)
                    cond_entry[v] = {
                        "target": rnd(cos[:, tgt].cpu()),
                        "nullmean": rnd(null_mean.cpu()),
                        "nullstd": rnd(null_std.cpu()),
                    }
                lat_file = latents_dir / f"{word}.json"
                if lat_file.exists():
                    sel = json.loads(lat_file.read_text())["layers"]
                    sae_vals, sae_meta = [], []
                    for layer in SAE_LAYERS:
                        idxs = [e["latent"] for e in sel.get(str(layer), [])]
                        sae_meta.append({
                            "layer": layer,
                            "latents": [
                                {"latent": e["latent"], "label": e["label"]}
                                for e in sel.get(str(layer), [])
                            ],
                        })
                        if idxs:
                            f = saes[layer].encode(A[layer].to(saes[layer].dtype))
                            sae_vals.append(
                                [[round(float(x), 2) for x in f[:, j]] for j in idxs])
                        else:
                            sae_vals.append(None)
                    cond_entry["sae"] = sae_vals
                    slot.setdefault("sae_latents", sae_meta)
            slot["conditions"][rec["condition"]] = cond_entry
        if (i + 1) % 40 == 0:
            print(f"{i + 1}/{len(records)}")

    payload = {
        "run_id": cfg.run_id,
        "sae_layers": SAE_LAYERS,
        "n_layers": 62,
        "variants": list(VARIANTS),
        "data": data,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    gz = gzip.compress(raw, 9)
    b64 = base64.b64encode(gz).decode()

    template = Path("viz/repr_viewer.html").read_text()
    assert "__DATA_B64__" in template
    out = run_dir / "results" / "repr_viewer.html"
    # The template is a headless body fragment (Artifact-friendly). For a
    # standalone file opened over file://, the browser gets no charset hint and
    # falls back to Windows-1252, mangling UTF-8 (± -> "Â±", — -> "â€""). Wrap
    # it in a minimal document declaring UTF-8. doctype + meta come before any
    # body content, so the template's <title>/<style> still parse into <head>.
    doc = '<!doctype html>\n<meta charset="utf-8">\n' + template.replace("__DATA_B64__", b64)
    out.write_text(doc, encoding="utf-8")
    print(f"json {len(raw) / 1e6:.1f} MB -> gzip {len(gz) / 1e6:.1f} MB; "
          f"viewer written to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(tyro.cli(Config))
