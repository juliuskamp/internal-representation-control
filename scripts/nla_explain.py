"""Decode stored response-token activations into NLA explanations.

Feeds layer-41 resid_post vectors from a run's `acts/*.pt` through the NLA
actor (activation verbalizer) `kitft/nla-gemma3-27b-L41-av`, served by a
patched SGLang server (see notes/nla_setup.md). Use it to compare what the
model is "thinking about" across conditions (think / dont_think / no_mention).

    uv run python scripts/nla_explain.py --run-id run1-core --words Dust \
        --conditions think dont_think no_mention --limit 3

Requires the SGLang server to be up (default http://localhost:30000). The
NLAClient itself only needs the checkpoint's tokenizer + embedding table on
CPU — no GPU in this process.

Layer indexing: the NLA actor was trained on the output of decoder block 41
(resid_post), which is exactly `acts[41]` in our capture (`ResidualCapture`
hooks decoder-layer outputs; see irc/model.py). Stored acts cover response
tokens only, so there is no BOS position to exclude.
"""

from irc import env  # noqa: F401  (must be first: loads .env, sets HF_HOME)

import argparse
import json
from pathlib import Path

import torch

from irc.vendor.nla_inference import NLAClient

REPO_ROOT = Path(__file__).resolve().parent.parent

AV_CHECKPOINT = Path(
    "/workspace/hf-cache/hub/models--kitft--nla-gemma3-27b-L41-av/snapshots/"
    "4e721238131ffb8348cff260fe81b8b34a270a0d"
)
NLA_LAYER = 41  # extraction_layer_index from nla_meta.yaml


def iter_generations(run_dir: Path, words: list[str] | None, conditions: list[str]):
    with open(run_dir / "generations.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["acts_file"] is None or not row["exact_match"]:
                continue
            if row["condition"] not in conditions:
                continue
            if words and row["condition"] != "no_mention" and row["word"] not in words:
                continue
            yield row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--words", nargs="*", default=None,
                    help="Concept words (Capitalized). Default: all.")
    ap.add_argument("--conditions", nargs="*",
                    default=["think", "dont_think", "no_mention"])
    ap.add_argument("--sentences", nargs="*", type=int, default=None,
                    help="Restrict to these sentence indices.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max generations per (word, condition).")
    ap.add_argument("--agg", choices=["mean", "token"], default="mean",
                    help="mean: one vector per generation (mean over response "
                         "tokens). token: one decode per response token.")
    ap.add_argument("--layer", type=int, default=NLA_LAYER,
                    help="Resid_post layer to decode (NLA is trained on 41; "
                         "other layers are OOD for the actor).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 (greedy) keeps the pipeline deterministic and is "
                         "upstream's reference mode; use 1.0 (the RL rollout "
                         "distribution) for repeat-sampling analyses.")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--out", default=None,
                    help="Output jsonl (default: results/nla_explanations_"
                         "{agg}_L{layer}.jsonl in the run dir). Appends.")
    args = ap.parse_args()

    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id
    out_path = Path(args.out) if args.out else (
        run_dir / "results" / f"nla_explanations_{args.agg}_L{args.layer}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = NLAClient(AV_CHECKPOINT, sglang_url=args.sglang_url)

    per_slot: dict[tuple[str, str], int] = {}
    n_done = 0
    with open(out_path, "a") as out:
        for row in iter_generations(run_dir, args.words, args.conditions):
            if args.sentences and row["sentence_idx"] not in args.sentences:
                continue
            slot = (row["condition"], row["word"] or "NONE")
            per_slot[slot] = per_slot.get(slot, 0)
            if args.limit is not None and per_slot[slot] >= args.limit:
                continue
            per_slot[slot] += 1

            acts = torch.load(run_dir / row["acts_file"], map_location="cpu")
            layer_acts = acts[args.layer].float()  # [tokens, d_model]
            n_sent = len(client.tokenizer(
                row["sentence"], add_special_tokens=False)["input_ids"])
            assert layer_acts.shape[0] >= n_sent, \
                f"{row['key']}: acts shorter than sentence"
            layer_acts = layer_acts[:n_sent]  # drop trailing whitespace token(s)

            if args.agg == "mean":
                vectors = [("mean", layer_acts.mean(dim=0))]
            else:
                vectors = [(t, layer_acts[t]) for t in range(layer_acts.shape[0])]

            for pos, v in vectors:
                explanation = client.generate(
                    v,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
                rec = {
                    "key": row["key"],
                    "condition": row["condition"],
                    "word": row["word"],
                    "sentence_idx": row["sentence_idx"],
                    "layer": args.layer,
                    "position": pos,
                    "norm": v.norm().item(),
                    "explanation": explanation,
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_done += 1
                print(f"[{n_done}] {row['key']} pos={pos} "
                      f"norm={rec['norm']:.0f}\n    {explanation[:160]}")

    print(f"\nwrote {n_done} explanations to {out_path}")


if __name__ == "__main__":
    main()
