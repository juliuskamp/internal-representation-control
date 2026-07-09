"""Export per-token representation-strength data as chunked files for the
GitHub-Pages viewer (docs/index.html).

For every stored (word, sentence, condition) run of a pipeline run:
  - concept-vector cosines per layer x token for the target word (both variants)
  - control-word null band (mean ± std across the 100 control words) per layer x
    token, computed per condition. The no_mention band is word-independent
    (shared activations x control vectors), so it is stored once per sentence
    in shared-bands.json.gz instead of being duplicated into every word chunk.
  - SAE activation per layer x selected latent x token (aggregation and latent
    toggling happen client-side in the viewer)
  - NLA explanations per token (layer 41), when results/nla_explanations_
    token_L41.jsonl exists in the run dir (written by scripts/nla_explain.py
    --agg token): the explanation text plus the LLM-judge score (0-100 logit
    expectation, from results/nla_judgments_token_L41.jsonl, written by
    scripts/nla_judge.py) and the judge's evidence quote per token.
    no_mention explanations are word-independent; they are attached (with
    that word's judgments) per word, but only for words that have their own
    NLA rows on that sentence. Tokens without a judgment get null scores.
All series are trimmed to the sentence's own tokens (the model sometimes emits a
trailing whitespace token before <end_of_turn>; verified to be the only length
mismatch, and responses always begin directly with the sentence tokens).

Writes (all fetched by the viewer, so docs/ must be served over HTTP):
  docs/data/index.json            run metadata + word list
  docs/data/shared-bands.json.gz  per-sentence no_mention null bands
  docs/data/words/{word}.json.gz  per-word slots (lazy-loaded on word change)

Usage: uv run python scripts/export_viz_data.py --run-id run1-core
View:  python -m http.server -d docs
"""

from irc import env  # noqa: F401

import dataclasses
import gzip
import json
from pathlib import Path

import torch
import tyro

from irc.constants import N_LAYERS, NLA_LAYER, SAE_LAYERS
from irc.constants import VECTOR_VARIANTS as VARIANTS
from irc.paths import ARTIFACTS, DOCS_DATA, RUNS
from irc.pipeline import _load_records, _load_saes
from irc.words_paper import CONTROL_WORDS_PAPER

DATA_DIR = DOCS_DATA


def load_nla(run_dir: Path) -> dict:
    """(condition, word|None, sentence_idx) -> {token_pos: explanation}."""
    path = run_dir / "results" / f"nla_explanations_token_L{NLA_LAYER}.jsonl"
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["layer"] != NLA_LAYER or not isinstance(r["position"], int):
            continue
        out.setdefault((r["condition"], r["word"], r["sentence_idx"]), {})[
            r["position"]] = r["explanation"]
    return out


def load_judgments(run_dir: Path) -> dict:
    """(condition, word|None, sentence_idx, target_word) -> {token_pos: row}."""
    path = run_dir / "results" / f"nla_judgments_token_L{NLA_LAYER}.jsonl"
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["layer"] != NLA_LAYER or not isinstance(r["position"], int):
            continue
        out.setdefault(
            (r["condition"], r["word"], r["sentence_idx"], r["target_word"]), {}
        )[r["position"]] = r
    return out


def nla_entry(expl_by_pos: dict, judg_by_pos: dict, n_tokens: int) -> dict:
    """Trim to sentence tokens; attach judge score (0-100 logit expectation)
    and evidence quote per token (null where unjudged)."""
    judg = [judg_by_pos.get(t) for t in range(n_tokens)]
    return {
        "explanations": [expl_by_pos.get(t, "") for t in range(n_tokens)],
        "score": [
            None if j is None or j["score_expected"] is None
            else round(j["score_expected"], 1)
            for j in judg
        ],
        "evidence": [j["evidence"] if j else None for j in judg],
    }


@dataclasses.dataclass
class Config:
    run_id: str = "run1-core"


def rnd(t: torch.Tensor, nd: int = 3) -> list:
    return [[round(float(x), nd) for x in row] for row in t]


def write_gz(path: Path, obj) -> tuple[int, int]:
    raw = json.dumps(obj, separators=(",", ":")).encode()
    path.write_bytes(gzip.compress(raw, 9, mtime=0))  # mtime=0: deterministic output
    return len(raw), path.stat().st_size


@torch.no_grad()
def main(cfg: Config) -> None:
    run_dir = RUNS / cfg.run_id
    records = _load_records(run_dir)

    from irc.model import load_tokenizer

    tokenizer = load_tokenizer()

    banks = {}
    for v in VARIANTS:
        bank = torch.load(ARTIFACTS / "concept_vectors" / f"bank_{v}_v1.pt")
        words = list(bank["vectors"].keys())
        V = torch.stack([bank["vectors"][w] for w in words]).cuda()
        banks[v] = {
            "w_idx": {w: i for i, w in enumerate(words)},
            "ctrl_idx": torch.tensor([i for i, w in enumerate(words) if w in set(CONTROL_WORDS_PAPER)]),
            "Vn": V / V.norm(dim=-1, keepdim=True),
        }

    saes = _load_saes(SAE_LAYERS)
    latents_dir = ARTIFACTS / "latents_v1"
    sel_cache: dict[str, dict | None] = {}

    nla = load_nla(run_dir)
    judgments = load_judgments(run_dir)
    # (word, si) pairs with their own NLA rows — gates attaching the shared
    # no_mention explanations so they aren't duplicated into all 50 words.
    nla_worded = {(w, si) for (_, w, si) in nla if w is not None}
    judge_meta = None
    if judgments:
        any_row = next(iter(next(iter(judgments.values())).values()))
        judge_meta = {"model": any_row["judge_model"],
                      "prompt_version": any_row["prompt_version"]}

    def sel_for(word: str) -> dict | None:
        if word not in sel_cache:
            p = latents_dir / f"{word}.json"
            sel_cache[word] = json.loads(p.read_text())["layers"] if p.exists() else None
        return sel_cache[word]

    tokens_cache: dict[str, list[str]] = {}
    data: dict = {}   # word -> si -> slot
    shared: dict = {}  # si -> variant -> {nullmean, nullstd}
    for i, rec in enumerate(records):
        sk = str(rec["sentence_idx"])
        # word-free conditions (no_mention) apply to every word sharing the sentence
        target_words = (
            [rec["word"]] if rec["word"] else
            sorted({r["word"] for r in records if r["word"] and r["sentence_idx"] == rec["sentence_idx"]})
        )
        if sk not in tokens_cache:
            tokens_cache[sk] = [
                t.replace("▁", " ")
                for t in tokenizer.convert_ids_to_tokens(
                    tokenizer(rec["sentence"], add_special_tokens=False)["input_ids"])
            ]
        toks = tokens_cache[sk]

        cos_v = band_v = feats = None
        if rec["acts_file"]:
            A = torch.load(run_dir / rec["acts_file"]).float().cuda()  # (L,T,D)
            assert A.shape[1] >= len(toks), f"{rec['key']}: acts shorter than sentence"
            A = A[:, : len(toks)]  # drop trailing whitespace token(s)
            An = A / A.norm(dim=-1, keepdim=True)
            cos_v, band_v = {}, {}
            for v in VARIANTS:
                cos = torch.einsum("ltd,wld->lwt", An, banks[v]["Vn"])  # (L,W,T)
                cos_v[v] = cos
                cc = cos[:, banks[v]["ctrl_idx"]]  # (L, n_control, T)
                band = {"nullmean": rnd(cc.mean(dim=1).cpu()), "nullstd": rnd(cc.std(dim=1).cpu())}
                if rec["condition"] == "no_mention":
                    shared.setdefault(sk, {}).setdefault(v, band)
                else:
                    band_v[v] = band
            feats = {l: saes[l].encode(A[l].to(saes[l].dtype)) for l in SAE_LAYERS}

        for word in target_words:
            slot = data.setdefault(word, {}).setdefault(sk, {
                "sentence": rec["sentence"], "tokens": toks, "conditions": {},
            })
            entry: dict = {"exact": rec["exact_match"], "completion": rec["completion"]}
            expl_by_pos = nla.get((rec["condition"], rec["word"], rec["sentence_idx"]))
            if expl_by_pos and (word, rec["sentence_idx"]) in nla_worded:
                judg_by_pos = judgments.get(
                    (rec["condition"], rec["word"], rec["sentence_idx"], word), {})
                entry["nla"] = nla_entry(expl_by_pos, judg_by_pos, len(toks))
            if cos_v is not None:
                for v in VARIANTS:
                    e = {"target": rnd(cos_v[v][:, banks[v]["w_idx"][word]].cpu())}
                    e.update(band_v.get(v, {}))
                    entry[v] = e
                sel = sel_for(word)
                if sel is not None:
                    sae_vals, sae_meta = [], []
                    for l in SAE_LAYERS:
                        es = sel.get(str(l), [])
                        sae_meta.append({
                            "layer": l,
                            "latents": [{"latent": x["latent"], "label": x["label"]} for x in es],
                        })
                        if es:
                            f_sel = feats[l][:, [x["latent"] for x in es]].float().cpu()
                            sae_vals.append([[round(x, 2) for x in col] for col in f_sel.t().tolist()])
                        else:
                            sae_vals.append(None)
                    entry["sae"] = sae_vals
                    slot.setdefault("sae_latents", sae_meta)
            slot["conditions"][rec["condition"]] = entry
        if (i + 1) % 40 == 0:
            print(f"{i + 1}/{len(records)}")

    words_dir = DATA_DIR / "words"
    words_dir.mkdir(parents=True, exist_ok=True)
    for stale in words_dir.glob("*.json.gz"):
        stale.unlink()
    total_raw = total_gz = 0
    for word, sis in sorted(data.items()):
        r, g = write_gz(words_dir / f"{word}.json.gz", {"word": word, "slots": sis})
        total_raw += r; total_gz += g
    r, g = write_gz(DATA_DIR / "shared-bands.json.gz", shared)
    total_raw += r; total_gz += g

    # Per (word, sentence) exactness, for greying out dropdown options that
    # have no usable data: "exact" means both word-conditioned generations
    # (think, dont_think) hit the target sentence exactly. no_mention is
    # word-independent so it's excluded from this check. Encoded as one
    # bitstring per word, aligned to the shared sentence_order list (every
    # word has the same sentence set).
    sentence_order = sorted({si for sis in data.values() for si in sis}, key=int)
    exact_by_word = {}
    for word, sis in data.items():
        bits = []
        for si in sentence_order:
            slot = sis.get(si)
            conds = slot["conditions"] if slot else {}
            ok = all((conds.get(c) or {}).get("exact") for c in ("think", "dont_think"))
            bits.append("1" if ok else "0")
        exact_by_word[word] = "".join(bits)

    (DATA_DIR / "index.json").write_text(json.dumps({
        "run_id": cfg.run_id,
        "sae_layers": list(SAE_LAYERS),
        "n_layers": N_LAYERS,
        "variants": list(VARIANTS),
        "nla_layer": NLA_LAYER,
        "nla_judge": judge_meta,
        "words": sorted(data),
        "sentence_order": sentence_order,
        "exact": exact_by_word,
    }, indent=1))
    print(f"{len(data)} word chunks + shared bands: "
          f"json {total_raw / 1e6:.1f} MB -> gzip {total_gz / 1e6:.1f} MB in {DATA_DIR}")


if __name__ == "__main__":
    main(tyro.cli(Config))
