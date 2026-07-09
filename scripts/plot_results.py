"""Figures for an intentional-control run (paper Fig. 24/26 analogues).

Usage: uv run python scripts/plot_results.py --run-id run1-core
Outputs PNGs into artifacts/runs/{run_id}/results/figures/.
"""

from irc import env  # noqa: F401

import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tyro
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from irc.paths import RUNS

# Reference categorical palette (dataviz skill), fixed slot order.
COND_COLOR = {"think": "#2a78d6", "dont_think": "#eda100", "no_mention": "#1baf7a"}
COND_LABEL = {"think": "think", "dont_think": "don't think", "no_mention": "no mention"}
NULL_GRAY = "#b9b7b0"
TEXT_SECONDARY = "#52514e"
DIVERGING = LinearSegmentedColormap.from_list(
    "irc_div", ["#2a78d6", "#f4f4f2", "#e34948"]
)


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


@dataclasses.dataclass
class Config:
    run_id: str
    heatmap_examples: int = 4  # per-token heatmaps for the first N think-pairs
    heatmap_layer_variant: str = "paper"


def null_band(run_dir: Path, variant: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Paper-style band: q05/q95 across control words of the pair-averaged
    token-mean cosine (computed on the think-condition runs)."""
    files = sorted((run_dir / "results" / "null_means").glob(f"{variant}__think__*.pt"))
    if not files:
        return None
    stack = torch.stack([torch.load(f).float() for f in files])  # (runs, L, C)
    per_control = stack.mean(0)  # (L, C) mean over pairs
    q = torch.quantile(per_control, torch.tensor([0.05, 0.95]), dim=1)
    return q[0].numpy(), q[1].numpy()


def layer_curves(df: pd.DataFrame, run_dir: Path, fig_dir: Path) -> None:
    for variant, vdf in df.groupby("variant"):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        g = vdf.groupby(["condition", "layer"])["cos_mean"].mean().reset_index()
        band = null_band(run_dir, variant)
        if band is not None:
            layers = np.arange(len(band[0]))
            ax.fill_between(layers, band[0], band[1],
                            color=NULL_GRAY, alpha=0.45, linewidth=0,
                            label="control words (5–95%)")
        else:  # fallback: per-run quantiles averaged over pairs (wider band)
            null = vdf.groupby("layer")[["null_q05", "null_q95"]].mean()
            ax.fill_between(null.index, null["null_q05"], null["null_q95"],
                            color=NULL_GRAY, alpha=0.45, linewidth=0,
                            label="control words (5–95%, per-run)")
        for cond in ("think", "dont_think", "no_mention"):
            sub = g[g["condition"] == cond]
            if sub.empty:
                continue
            ax.plot(sub["layer"], sub["cos_mean"], color=COND_COLOR[cond],
                    linewidth=2, label=COND_LABEL[cond])
        ax.set_xlabel("layer")
        ax.set_ylabel("cosine with target-word concept vector\n(mean over response tokens and pairs)")
        ax.set_title(f"Concept representation strength by condition — {variant} vectors")
        ax.legend(frameon=False, loc="upper left")
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f"layer_curves_{variant}.png", dpi=200)
        plt.close(fig)


def delta_curves(df: pd.DataFrame, fig_dir: Path) -> None:
    """think minus don't-think gap per layer, both variants on small multiples."""
    variants = sorted(df["variant"].unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(7.2, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, variant in zip(axes, variants):
        vdf = df[df["variant"] == variant]
        m = vdf.pivot_table(index="layer", columns="condition", values="cos_mean")
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        if {"think", "dont_think"} <= set(m.columns):
            ax.plot(m.index, m["think"] - m["dont_think"], color="#4a3aa7",
                    linewidth=2, label="think − don't think")
        if {"think", "no_mention"} <= set(m.columns):
            ax.plot(m.index, m["think"] - m["no_mention"], color="#e87ba4",
                    linewidth=2, label="think − no mention")
        ax.set_title(f"{variant} vectors", fontsize=10)
        ax.set_xlabel("layer")
        style_axis(ax)
    axes[0].set_ylabel("Δ cosine")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Condition gaps across layers", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "delta_curves.png", dpi=200)
    plt.close(fig)


def token_heatmaps(run_dir: Path, cfg: Config, fig_dir: Path) -> None:
    from irc.model import load_tokenizer

    tokenizer = load_tokenizer()
    records = {}
    with (run_dir / "generations.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            records[r["key"]] = r
    tc_dir = run_dir / "results" / "token_cosines"
    think_files = sorted(tc_dir.glob(f"{cfg.heatmap_layer_variant}__think__*.pt"))
    for path in think_files[: cfg.heatmap_examples]:
        # filenames are f"{variant}__{record_key}.pt" (see pipeline.measure)
        key = path.stem.removeprefix(f"{cfg.heatmap_layer_variant}__")
        rec = records[key]
        cos = torch.load(path).numpy()  # (layers, tokens)
        ids = tokenizer(rec["sentence"], add_special_tokens=False)["input_ids"]
        toks = [t.replace("▁", " ") for t in tokenizer.convert_ids_to_tokens(ids)]
        fig, ax = plt.subplots(figsize=(0.55 * len(toks) + 2.2, 4.6))
        vmax = max(abs(cos).max(), 1e-6)
        im = ax.imshow(cos, aspect="auto", cmap=DIVERGING.reversed(),
                       norm=TwoSlopeNorm(0, -vmax, vmax), origin="lower")
        ax.set_xticks(range(len(toks)), toks, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("layer")
        ax.set_title(f'"{rec["word"]}" while writing s{rec["sentence_idx"]:02d} (think)',
                     fontsize=10)
        fig.colorbar(im, ax=ax, label="cosine", shrink=0.85)
        fig.tight_layout()
        fig.savefig(fig_dir / f"tokens_{key}.png", dpi=200)
        plt.close(fig)


def sae_curves(run_dir: Path, fig_dir: Path) -> None:
    path = run_dir / "results" / "sae_latents.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for cond in ("think", "dont_think", "no_mention"):
        sub = df[df["condition"] == cond].groupby("layer")["act_sum_mean"].mean()
        if sub.empty:
            continue
        ax.plot(sub.index, sub.values, color=COND_COLOR[cond], linewidth=2,
                marker="o", markersize=6, label=COND_LABEL[cond])
    ax.set_xticks(sorted(df["layer"].unique()))
    ax.set_xlabel("layer (Gemma Scope 2, 16k l0_medium)")
    ax.set_ylabel("selected-latent activation\n(sum over top-5 latents, mean over tokens & pairs)")
    ax.set_title("SAE latent evidence by condition")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "sae_curves.png", dpi=200)
    plt.close(fig)


def main(cfg: Config) -> None:
    run_dir = RUNS / cfg.run_id
    fig_dir = run_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(run_dir / "results" / "concept_cosines.parquet")
    layer_curves(df, run_dir, fig_dir)
    delta_curves(df, fig_dir)
    token_heatmaps(run_dir, cfg, fig_dir)
    sae_curves(run_dir, fig_dir)
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
