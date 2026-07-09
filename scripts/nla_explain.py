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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import os

import torch

from irc.constants import NLA_LAYER, NLA_REPO
from irc.paths import RUNS
from irc.vendor.nla_inference import _EMBED_KEY_SUFFIXES, NLAClient

# Everything NLAClient reads except the weight shards: tokenizer, configs,
# nla_meta.yaml, and the safetensors index (which names the embedding shard).
_META_PATTERNS = ["*.json", "*.yaml", "tokenizer*", "*.model",
                  "special_tokens*", "added_tokens*", "chat_template*"]


def resolve_checkpoint() -> Path:
    """Local snapshot dir of the NLA actor checkpoint, via the HF cache.

    Downloads only what NLAClient reads (tokenizer + configs + the one shard
    holding embed_tokens, ~2 GB) — never the full ~50 GB checkpoint. Prefers
    the cache (offline-safe); set NLA_AV_CHECKPOINT in .env to point at an
    existing local snapshot outside HF_HOME instead.
    """
    if override := os.environ.get("NLA_AV_CHECKPOINT"):
        return Path(override)
    from huggingface_hub import snapshot_download

    def snap(patterns: list[str]) -> Path:
        try:
            return Path(snapshot_download(
                NLA_REPO, allow_patterns=patterns, local_files_only=True))
        except Exception:
            return Path(snapshot_download(NLA_REPO, allow_patterns=patterns))

    root = snap(_META_PATTERNS)
    weight_map = json.loads(
        (root / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({f for k, f in weight_map.items()
                     if k.endswith(_EMBED_KEY_SUFFIXES)})
    snap(_META_PATTERNS + shards)
    return root


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
    ap.add_argument("--concurrency", type=int, default=8,
                    help="In-flight SGLang requests (server-side continuous "
                         "batching packs them). 1 = sequential.")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--out", default=None,
                    help="Output jsonl (default: results/nla_explanations_"
                         "{agg}_L{layer}.jsonl in the run dir). Appends.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-decode everything. Default resumes: (key, "
                         "position) pairs already in the output file are "
                         "skipped, so an interrupted run continues where it "
                         "stopped instead of appending duplicates.")
    args = ap.parse_args()

    run_dir = RUNS / args.run_id
    out_path = Path(args.out) if args.out else (
        run_dir / "results" / f"nla_explanations_{args.agg}_L{args.layer}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = NLAClient(resolve_checkpoint(), sglang_url=args.sglang_url)

    # Resume: collect (key, position) pairs already in the output file so an
    # interrupted run continues instead of appending duplicates. Tolerates a
    # truncated final line from a killed process.
    done: set[tuple[str, object]] = set()
    if not args.no_resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line from an interrupted write
                done.add((rec["key"], rec["position"]))
        if done:
            print(f"[resume] {len(done)} explanations already in {out_path.name}"
                  f" — skipping those")

    def iter_row_positions():
        """Yield (row, positions) for rows passing the filters, where positions
        is the list of position ids still to decode (after resume-skipping).
        Does NOT load acts: the token-mode position count comes from tokenizing
        the sentence, which is exactly what decode truncates the acts to (the
        `layer_acts.shape[0] >= n_sent` assert below guarantees it). Cheap
        enough to run twice — once to count, once to drive generation."""
        per_slot: dict[tuple[str, str], int] = {}
        for row in iter_generations(run_dir, args.words, args.conditions):
            if args.sentences and row["sentence_idx"] not in args.sentences:
                continue
            slot = (row["condition"], row["word"] or "NONE")
            per_slot[slot] = per_slot.get(slot, 0)
            if args.limit is not None and per_slot[slot] >= args.limit:
                continue
            per_slot[slot] += 1

            if args.agg == "mean":
                positions: list[object] = ["mean"]
            else:
                n_sent = len(client.tokenizer(
                    row["sentence"], add_special_tokens=False)["input_ids"])
                positions = list(range(n_sent))
            positions = [p for p in positions if (row["key"], p) not in done]
            if positions:  # skip fully-done rows (no acts load happens for them)
                yield row, positions

    # Cheap first pass over the filters to size the progress bar. Only tokenizes
    # sentences (no acts loaded), so it's fast even in token mode.
    total = sum(len(positions) for _, positions in iter_row_positions())
    if total == 0:
        print("nothing to decode — all requested tasks are already done "
              "(or filtered out). Pass --no-resume to redo.")
        return
    print(f"decoding {total} explanation(s), concurrency={args.concurrency}")

    def iter_tasks():
        """Yield (row, pos, vector) tasks. Loads acts lazily (once per row, and
        only for rows with remaining positions) so disk I/O overlaps with
        in-flight generation rather than blocking startup."""
        for row, positions in iter_row_positions():
            acts = torch.load(run_dir / row["acts_file"], map_location="cpu")
            layer_acts = acts[args.layer].float()  # [tokens, d_model]
            n_sent = len(client.tokenizer(
                row["sentence"], add_special_tokens=False)["input_ids"])
            assert layer_acts.shape[0] >= n_sent, \
                f"{row['key']}: acts shorter than sentence"
            layer_acts = layer_acts[:n_sent]  # drop trailing whitespace token(s)

            if args.agg == "mean":
                vecs = {"mean": layer_acts.mean(dim=0)}
            else:
                vecs = {t: layer_acts[t] for t in range(layer_acts.shape[0])}
            for pos in positions:
                yield row, pos, vecs[pos]

    def decode(task):
        """Runs on a worker thread: blocking HTTP round-trip to SGLang. The
        server's continuous batcher packs concurrent requests onto the GPU."""
        row, pos, v = task
        explanation = client.generate(
            v, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
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
        return row, pos, rec

    # Bounded producer/consumer: keep ~2×concurrency requests in flight so the
    # GPU batcher stays fed while the main thread loads the next acts + writes
    # completed rows. Results are written as they finish (order is not
    # significant — each jsonl row is self-describing via `key`/`position`).
    n_done = 0
    tasks = iter_tasks()
    max_inflight = max(1, args.concurrency) * 2
    with open(out_path, "a") as out, \
            ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        inflight = set()
        for _ in range(max_inflight):
            try:
                inflight.add(ex.submit(decode, next(tasks)))
            except StopIteration:
                break
        while inflight:
            finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                row, pos, rec = fut.result()
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_done += 1
                print(f"[{n_done}/{total} {100 * n_done // total}%] "
                      f"{rec['key']} pos={pos} norm={rec['norm']:.0f}\n"
                      f"    {rec['explanation'][:160]}")
                try:
                    inflight.add(ex.submit(decode, next(tasks)))
                except StopIteration:
                    pass

    print(f"\nwrote {n_done} explanations to {out_path}")


if __name__ == "__main__":
    main()
