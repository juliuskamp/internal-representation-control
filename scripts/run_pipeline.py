"""Run the intentional-control replication pipeline.

Examples:
    uv run python scripts/run_pipeline.py --run-id run1
    uv run python scripts/run_pipeline.py --run-id run1 --stages generate measure
    uv run python scripts/run_pipeline.py --run-id run1 --sentences-per-word 5
"""

from irc import env  # noqa: F401

import dataclasses
import json
import subprocess
import time
from pathlib import Path

import tyro

from irc import pipeline
from irc.words_paper import CONCEPT_WORDS_PAPER


@dataclasses.dataclass
class Config:
    run_id: str = dataclasses.field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))
    stages: tuple[str, ...] = ("vectors", "generate", "latents", "measure")
    seed: int = 0
    sentences_per_word: int = 2
    conditions: tuple[str, ...] = ("think", "dont_think", "no_mention")
    words: tuple[str, ...] = ()  # empty = all 50 paper concept words
    vector_variants: tuple[str, ...] = ("paper", "word_tokens")
    sae_layers: tuple[int, ...] = (16, 31, 40, 53)
    topk_latents: int = 5
    neuronpedia: bool = True


def main(cfg: Config) -> None:
    run_dir = Path("artifacts/runs") / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    import sae_lens, torch, transformers  # noqa: E401

    from irc.model import MODEL_ID

    (run_dir / "config.json").write_text(json.dumps({
        **dataclasses.asdict(cfg),
        "model_id": MODEL_ID,
        "sae_release": pipeline.SAE_RELEASE,
        "sae_id_template": pipeline.SAE_ID_TEMPLATE,
        "git_commit": commit,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sae_lens": sae_lens.__version__,
        "word_lists": "irc/words_paper.py (baseline deduplicated to 99)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1))

    words = list(cfg.words) or CONCEPT_WORDS_PAPER
    pairs = pipeline.pair_table(words, cfg.sentences_per_word, cfg.seed)
    print(f"run {cfg.run_id}: {len(words)} words x {cfg.sentences_per_word} sentences, "
          f"conditions={cfg.conditions}, stages={cfg.stages}")

    model = tokenizer = None
    if {"vectors", "generate", "latents"} & set(cfg.stages):
        from irc.model import load_model, load_tokenizer

        tokenizer = load_tokenizer()
        model = load_model()

    if "vectors" in cfg.stages:
        for variant in cfg.vector_variants:
            pipeline.ensure_vector_bank(model, tokenizer, variant)
    if "generate" in cfg.stages:
        pipeline.run_generations(model, tokenizer, run_dir, pairs, list(cfg.conditions))
    if "latents" in cfg.stages:
        pipeline.select_latents(
            model, tokenizer, words, list(cfg.sae_layers), cfg.topk_latents, cfg.neuronpedia
        )
    if "measure" in cfg.stages:
        del model  # free VRAM for SAE encodes
        pipeline.measure(
            run_dir, pairs, list(cfg.conditions), list(cfg.vector_variants),
            list(cfg.sae_layers),
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
