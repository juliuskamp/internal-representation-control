"""Render one docs/ viewer chart as a static image (for slides / papers).

Reads the same exported data the browser viewers fetch (docs/data/, written by
scripts/export_viz_data.py + export_agg_data.py) and reproduces any of the four
pages' charts — no GPU or model needed, matching the interactive viewers in
style (palette, bands, direct end labels, rotated token labels). --chart picks
the page:

  word       docs/index.html      one word x sentence, tokens on x (default)
  aggregate  docs/aggregate.html  word-mean ±1 std for one sentence, tokens on x
  layers     docs/layers.html     token+sentence-collapsed, layers on x
  forest     docs/forest.html     per-word paired Δ at one layer, words on y

Usage:
  uv run python scripts/render_viewer_figure.py                    # list words
  uv run python scripts/render_viewer_figure.py --word Dust --sent 3
  uv run python scripts/render_viewer_figure.py --word Dust --sent 3 \
      --meas sae_v2 --layer 40 --agg sum --theme dark --out dust.svg
  uv run python scripts/render_viewer_figure.py --chart aggregate --sent 3 --delta
  uv run python scripts/render_viewer_figure.py --chart layers --delta --theme dark
  uv run python scripts/render_viewer_figure.py --chart forest --layer 40

Output format follows the --out extension (png/svg/pdf); default is a PNG in
artifacts/figures/. Excluded (non-exact) conditions are dropped from the chart
and reported on stdout, like the viewer's red chips; the matching page's footer
meta line (and for SAE charts the selected latents) is printed for use as a
slide caption.
"""

from irc import env  # noqa: F401

import dataclasses
import gzip
import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from irc.paths import ARTIFACTS, DOCS_DATA

# Mirror of the viewer palette in docs/index.html (light-dark() pairs).
THEMES = {
    "light": {
        "surface": "#fbfbf9", "ink": "#171614", "ink2": "#55534d",
        "line": "#e2e1db", "null": "#b9b7b0", "bad": "#9c2a29",
        "think": "#2a78d6", "dont_think": "#eda100", "no_mention": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f2f1ec", "ink2": "#c3c2b7",
        "line": "#35342f", "null": "#55534d", "bad": "#e8908f",
        "think": "#3987e5", "dont_think": "#c98500", "no_mention": "#199e70",
    },
}
COND = [("think", "think"), ("dont_think", "don't think"), ("no_mention", "no mention")]
MARKERS = {"think": "o", "dont_think": "s", "no_mention": "^"}
MONO = "DejaVu Sans Mono"

# Viewer chart geometry (px); fonts converted at 1px = 0.75pt.
W, H = 1020, 430
MARGIN = {"t": 16, "r": 28, "b": 86, "l": 58}
PT = 0.75


@dataclasses.dataclass
class Config:
    chart: Literal["word", "aggregate", "layers", "forest"] = "word"
    """Which viewer page to render (see module docstring)."""
    word: str | None = None
    """word chart: concept word (as in the viewer dropdown; case-insensitive).
    Omit to list the available words and sentences."""
    sent: int = 0
    """word/aggregate charts: sentence index (0-based, the sNN number in the
    viewer dropdown)."""
    meas: Literal["word_tokens", "paper", "sae", "sae_v2", "nla"] = "word_tokens"
    """Measurement: concept-vector variant, SAE latent selection, or NLA.
    layers/forest charts only have the concept-vector variants."""
    layer: int = 40
    """Residual layer (word/aggregate/forest charts); snapped to the nearest
    SAE layer for sae/sae_v2 and fixed to the NLA layer for nla."""
    base: Literal["band", "none"] = "band"
    """word chart baseline: control-word ±1 std bands per condition, or none.
    band is only available for concept-vector measurements (falls back to
    none otherwise)."""
    agg: Literal["sum", "mean", "max"] = "sum"
    """Aggregation over the selected SAE latents (sae/sae_v2 only)."""
    delta: bool = False
    """aggregate/layers charts: paired per-word Δ vs the no-mention condition
    instead of raw levels (the forest chart is always the paired Δ)."""
    complete: bool = True
    """aggregate/layers/forest charts: only words where all three conditions
    are exact per sentence (--no-complete for every exact pair)."""
    bands: bool = True
    """aggregate/layers charts: shaded ±1 std across words (--no-bands)."""
    sort: Literal["think", "dont", "alpha"] = "think"
    """forest chart row order: by think Δ, by don't-think Δ, or alphabetical."""
    theme: Literal["light", "dark"] = "light"
    title: str = ""
    """Optional title above the chart (default: none — slides have their own)."""
    out: Path | None = None
    """Output file; format from extension (.png/.svg/.pdf). Default: a PNG in
    artifacts/figures/ named after the selection."""
    dpi: int = 200
    transparent: bool = False
    """Transparent background instead of the theme surface color."""
    summary: bool = True
    """word/aggregate charts: summary column right of the chart — mean ±1 std
    over tokens (word) / of the per-word token-means (aggregate)
    (--no-summary to hide)."""


def load_gz(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def load_slot(index: dict, word: str, si: str) -> dict:
    chunk = load_gz(DOCS_DATA / "words" / f"{word}.json.gz")
    if si not in chunk["slots"]:
        raise SystemExit(f"sentence {si} not in data for {word} "
                         f"(have {', '.join(sorted(chunk['slots'], key=int))})")
    slot = chunk["slots"][si]
    # Splice the shared word-independent no_mention null bands into the slot,
    # exactly like the viewer's mergeShared().
    shared = load_gz(DOCS_DATA / "shared-bands.json.gz")
    nm, sh = slot["conditions"].get("no_mention"), shared.get(si)
    if nm and sh:
        for v in index["variants"]:
            if v in nm and v in sh:
                nm[v].update(sh[v])
    return slot


def aggregate_sae(rows: list[list[float]], agg: str) -> list[float]:
    a = np.asarray(rows, dtype=float)  # (n_latents, T)
    return {"sum": a.sum(0), "mean": a.mean(0), "max": a.max(0)}[agg].tolist()


def series_data(slot: dict, cfg: Config, layer: int, sae_layers: list[int]) -> list[dict]:
    """Port of the viewer's seriesData(): one entry per condition with vals
    (list[float | None] | None), band (lo, hi) | None, excluded, completion."""
    show_bands = cfg.base == "band" and cfg.meas not in ("sae", "sae_v2", "nla")
    out = []
    for cid, label in COND:
        rec = slot["conditions"].get(cid)
        c = {"id": cid, "label": label, "vals": None, "band": None,
             "excluded": False, "completion": None}
        if rec is None:
            out.append(c)
            continue
        c["excluded"], c["completion"] = not rec["exact"], rec["completion"]
        if rec["exact"]:
            if cfg.meas == "nla":
                nla = rec.get("nla")
                if nla and any(s is not None for s in nla["score"]):
                    c["vals"] = nla["score"]
            elif cfg.meas in ("sae", "sae_v2"):
                li = sae_layers.index(layer)
                rows = (rec.get(cfg.meas) or [None] * len(sae_layers))[li]
                if rows:
                    c["vals"] = aggregate_sae(rows, cfg.agg)
            elif rec.get(cfg.meas):
                e = rec[cfg.meas]
                c["vals"] = e["target"][layer]
                if show_bands and "nullmean" in e:
                    mean = np.asarray(e["nullmean"][layer])
                    std = np.asarray(e["nullstd"][layer])
                    c["band"] = (mean - std, mean + std)
        out.append(c)
    return out


def y_label(cfg: Config) -> str:
    if cfg.meas == "nla":
        return "judge score — concept in NLA explanation (0–100)"
    if cfg.meas in ("sae", "sae_v2"):
        agg = "average" if cfg.agg == "mean" else cfg.agg
        return f"selected-latent activation ({agg})"
    return "cosine with concept vector"


def draw(tokens: list[str], conds: list[dict], cfg: Config, pal: dict,
         out: Path) -> None:
    n = len(tokens)
    top = MARGIN["t"] + 26 + (24 if cfg.title else 0)  # room for the legend row
    h = H - MARGIN["t"] + top
    iw, ih = W - MARGIN["l"] - MARGIN["r"], H - MARGIN["t"] - MARGIN["b"]

    # per-condition mean ± std over the response tokens, for the summary panel
    summ = []
    if cfg.summary:
        for c in conds:
            if c["vals"] is None:
                continue
            v = np.array([np.nan if x is None else x for x in c["vals"]], dtype=float)
            if not np.isnan(v).all():
                summ.append((c, float(np.nanmean(v)), float(np.nanstd(v))))
    sw = 50 if summ else 0  # summary panel (36 px) + gap
    tw = W + sw

    fig = plt.figure(figsize=(tw / 96, h / 96), dpi=cfg.dpi)
    fig.patch.set_facecolor("none" if cfg.transparent else pal["surface"])
    ax = fig.add_axes([MARGIN["l"] / tw, MARGIN["b"] / h, iw / tw, ih / h])
    ax.set_facecolor("none")

    # y range like the viewer: anchored to include 0, 8% padding
    lo = hi = 0.0
    for c in conds:
        for v in c["vals"] or []:
            if v is not None:
                lo, hi = min(lo, v), max(hi, v)
        if c["band"] is not None:
            lo = min(lo, c["band"][0].min())
            hi = max(hi, c["band"][1].max())
    for _, m, s in summ:
        lo, hi = min(lo, m - s), max(hi, m + s)
    if hi == lo:
        hi = lo + 1
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)
    ax.set_xlim(0 if n > 1 else -0.5, n - 1 if n > 1 else 0.5)

    for side in ax.spines.values():
        side.set_visible(False)
    ticks = np.linspace(lo, hi, 6)
    fmt = (lambda v: f"{v:.2f}") if abs(hi - lo) < 5 else (lambda v: f"{v:.0f}")
    ax.set_yticks(ticks, [fmt(v) for v in ticks],
                  fontsize=11 * PT, color=pal["ink2"])
    ax.grid(axis="y", color=pal["line"], linewidth=0.75)
    ax.set_axisbelow(True)
    if lo < 0 < hi:
        ax.axhline(0, color=pal["ink2"], linewidth=0.75)

    ax.set_ylabel(y_label(cfg), fontsize=11.5 * PT, color=pal["ink2"], labelpad=8)

    if summ:
        # summary panel: an extra, separated "token" column at the right end
        # of the chart, on the same y axis as the lines
        sax = fig.add_axes([(MARGIN["l"] + iw + 14) / tw, MARGIN["b"] / h,
                            36 / tw, ih / h], sharey=ax)
        sax.set_facecolor("none")
        for side in sax.spines.values():
            side.set_visible(False)
        sax.tick_params(axis="y", labelleft=False, length=0)
        sax.grid(axis="y", color=pal["line"], linewidth=0.75)
        sax.set_axisbelow(True)
        if lo < 0 < hi:
            sax.axhline(0, color=pal["ink2"], linewidth=0.75)
        sax.set_xlim(-0.5, 0.5)
        sax.set_xticks([])
        sax.set_xlabel("mean\n±1 std", fontsize=10 * PT, color=pal["ink2"])
        k = len(summ)
        for i, (c, m, s) in enumerate(summ):
            color = pal[c["id"]]
            xi = (i - (k - 1) / 2) * (5 / 36)  # ~point-width x offsets
            sax.errorbar(xi, m, yerr=s, color=color, linewidth=2 * PT,
                         capsize=3, capthick=2 * PT)
            sax.plot(xi, m, linestyle="none", marker=MARKERS[c["id"]],
                     color=color, markersize=4.5, clip_on=False)

    ax.set_xticks(range(n), tokens, rotation=38, ha="right",
                  rotation_mode="anchor", fontsize=12 * PT,
                  fontfamily=MONO, color=pal["ink"])
    ax.tick_params(length=0)

    x = np.arange(n)
    for c in conds:  # bands behind the lines
        if c["band"] is not None:
            ax.fill_between(x, c["band"][0], c["band"][1],
                            color=pal[c["id"]], alpha=0.18, linewidth=0)

    for c in conds:
        if c["vals"] is None:
            continue
        color = pal[c["id"]]
        vals = np.array([np.nan if v is None else v for v in c["vals"]], dtype=float)
        ax.plot(x, vals, color=color, linewidth=2 * PT, solid_capstyle="round",
                solid_joinstyle="round", clip_on=False)
        ax.plot(x, vals, linestyle="none", marker=MARKERS[c["id"]], color=color,
                markersize=4.5, clip_on=False)

    # legend row above the chart, like the viewer's
    handles, labels = [], []
    for c in conds:
        if c["vals"] is None:
            continue
        handles.append(Line2D([], [], color=pal[c["id"]], linewidth=2 * PT,
                              marker=MARKERS[c["id"]], markersize=4.5))
        labels.append(c["label"])
    for c in conds:
        if c["band"] is not None:
            handles.append(Patch(facecolor=pal[c["id"]], alpha=0.5, linewidth=0))
            labels.append(f"{c['label']} baseline ±1 std")
    fig.legend(handles, labels, loc="upper left", frameon=False, ncol=len(handles),
               bbox_to_anchor=(MARGIN["l"] / tw, 1 - (MARGIN["t"] + (24 if cfg.title else 0)) / h),
               borderaxespad=0, fontsize=13 * PT, labelcolor=pal["ink2"],
               handlelength=1.4, columnspacing=1.6)
    if cfg.title:
        fig.text(MARGIN["l"] / tw, 1 - MARGIN["t"] / h, cfg.title,
                 fontsize=15 * PT, color=pal["ink"], va="top", fontweight="bold")
    excluded = [c for c in conds if c["excluded"]]
    if excluded:
        fig.text(1 - 14 / tw, 1 - MARGIN["t"] / h,
                 " · ".join(f"{c['label']}: excluded (non-exact)" for c in excluded),
                 fontsize=11 * PT, color=pal["bad"], va="top", ha="right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.dpi, transparent=cfg.transparent,
                facecolor=fig.get_facecolor())
    plt.close(fig)


CONDS_DELTA = [("think", "think"), ("dont_think", "don't think")]


def fmt_ticks(ax, lo: float, hi: float, pal: dict, n_ticks: int = 6) -> None:
    ticks = np.linspace(lo, hi, n_ticks)
    r = abs(hi - lo)
    fmt = (lambda v: f"{v:.3f}") if r < 0.5 else \
          (lambda v: f"{v:.2f}") if r < 5 else (lambda v: f"{v:.0f}")
    ax.set_yticks(ticks, [fmt(v) for v in ticks], fontsize=11 * PT, color=pal["ink2"])


def style_axis(ax, pal: dict, lo: float, hi: float) -> None:
    ax.set_facecolor("none")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(axis="y", color=pal["line"], linewidth=0.75)
    ax.set_axisbelow(True)
    if lo < 0 < hi:
        ax.axhline(0, color=pal["ink2"], linewidth=0.75)
    ax.tick_params(length=0)


def pad_range(lo: float, hi: float, frac: float = 0.08) -> tuple[float, float]:
    if hi == lo:
        hi = lo + 1
    pad = (hi - lo) * frac
    return lo - pad, hi + pad


def finish(fig, conds_legend: list[tuple], cfg: Config, pal: dict, out: Path,
           tw: float, h: float, extra_note: str = "") -> None:
    """Legend row + optional title, then save — shared by all charts."""
    handles = [Line2D([], [], color=col, linewidth=2 * PT, marker=mk,
                      markersize=4.5, linestyle=ls) for _, col, mk, ls in conds_legend]
    labels = [lb for lb, *_ in conds_legend]
    fig.legend(handles, labels, loc="upper left", frameon=False, ncol=len(handles),
               bbox_to_anchor=(MARGIN["l"] / tw, 1 - (MARGIN["t"] + (24 if cfg.title else 0)) / h),
               borderaxespad=0, fontsize=13 * PT, labelcolor=pal["ink2"],
               handlelength=1.4, columnspacing=1.6)
    if cfg.title:
        fig.text(MARGIN["l"] / tw, 1 - MARGIN["t"] / h, cfg.title,
                 fontsize=15 * PT, color=pal["ink"], va="top", fontweight="bold")
    if extra_note:
        fig.text(1 - 14 / tw, 1 - MARGIN["t"] / h, extra_note,
                 fontsize=11 * PT, color=pal["ink2"], va="top", ha="right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.dpi, transparent=cfg.transparent,
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---- aggregate chart (docs/aggregate.html) ---------------------------------

def agg_series(chunk: dict, cfg: Config, layer: int, sae_layers: list[int]) -> list[dict]:
    """Port of aggregate.html seriesData(): per condition, the word-mean/std
    token series {vals, stds, n, sm} for the current selection."""
    mode = "complete" if cfg.complete else "all"
    source = chunk["deltas"] if cfg.delta else chunk["conds"]
    conds = CONDS_DELTA if cfg.delta else COND
    out = []
    for cid, label in conds:
        block = (source.get(cid) or {}).get(mode)
        c = {"id": cid, "label": label, "vals": None, "stds": None,
             "n": None, "sm": None}
        if block:
            if cfg.meas == "nla":
                b = block.get("nla")
                if b:
                    c.update(vals=b["mean"], stds=b["std"], n=max(b["n"]),
                             sm=b.get("summary"))
            elif cfg.meas in ("sae", "sae_v2"):
                li = sae_layers.index(layer)
                vb = block.get(cfg.meas)
                if vb and vb[cfg.agg]["mean"][li]:
                    c.update(vals=vb[cfg.agg]["mean"][li],
                             stds=vb[cfg.agg]["std"][li], n=vb["n"][li])
                    s2 = vb[cfg.agg]["summary"]
                    if s2["mean"][li] is not None:
                        c["sm"] = {"mean": s2["mean"][li], "std": s2["std"][li]}
            elif block.get(cfg.meas):
                b = block[cfg.meas]
                c.update(vals=b["mean"][layer], stds=b["std"][layer], n=b["n"])
                if b.get("summary"):
                    c["sm"] = {"mean": b["summary"]["mean"][layer],
                               "std": b["summary"]["std"][layer]}
        out.append(c)
    return out


def draw_aggregate(tokens: list[str], conds: list[dict], cfg: Config,
                   pal: dict, out: Path) -> None:
    n = len(tokens)
    top = MARGIN["t"] + 26 + (24 if cfg.title else 0)
    h = H - MARGIN["t"] + top
    iw, ih = W - MARGIN["l"] - MARGIN["r"], H - MARGIN["t"] - MARGIN["b"]
    summ = [c for c in conds if c["sm"] and cfg.summary]
    sw = 50 if summ else 0
    tw = W + sw

    fig = plt.figure(figsize=(tw / 96, h / 96), dpi=cfg.dpi)
    fig.patch.set_facecolor("none" if cfg.transparent else pal["surface"])
    ax = fig.add_axes([MARGIN["l"] / tw, MARGIN["b"] / h, iw / tw, ih / h])

    lo = hi = 0.0
    for c in conds:
        for i, v in enumerate(c["vals"] or []):
            if v is None:
                continue
            lo, hi = min(lo, v), max(hi, v)
            if cfg.bands:
                lo = min(lo, v - c["stds"][i])
                hi = max(hi, v + c["stds"][i])
    for c in summ:
        lo = min(lo, c["sm"]["mean"] - c["sm"]["std"])
        hi = max(hi, c["sm"]["mean"] + c["sm"]["std"])
    lo, hi = pad_range(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlim(0 if n > 1 else -0.5, n - 1 if n > 1 else 0.5)
    style_axis(ax, pal, lo, hi)
    fmt_ticks(ax, lo, hi, pal)
    dp = "Δ vs no mention: " if cfg.delta else ""
    # labelpad 2, not 8: with the Δ prefix the rotated label otherwise pokes
    # past the left figure edge
    ax.set_ylabel(dp + "mean " + y_label(cfg),
                  fontsize=11.5 * PT, color=pal["ink2"], labelpad=2)
    ax.set_xticks(range(n), tokens, rotation=38, ha="right",
                  rotation_mode="anchor", fontsize=12 * PT,
                  fontfamily=MONO, color=pal["ink"])

    x = np.arange(n)
    for c in conds:  # ±1 std bands behind the lines
        if c["vals"] is None or not cfg.bands:
            continue
        v = np.array([np.nan if x_ is None else x_ for x_ in c["vals"]], dtype=float)
        s = np.array([np.nan if x_ is None else x_ for x_ in c["stds"]], dtype=float)
        ax.fill_between(x, v - s, v + s, color=pal[c["id"]], alpha=0.18, linewidth=0)
    for c in conds:
        if c["vals"] is None:
            continue
        v = np.array([np.nan if x_ is None else x_ for x_ in c["vals"]], dtype=float)
        ax.plot(x, v, color=pal[c["id"]], linewidth=2 * PT,
                solid_capstyle="round", solid_joinstyle="round", clip_on=False)
        ax.plot(x, v, linestyle="none", marker=MARKERS[c["id"]],
                color=pal[c["id"]], markersize=4.5, clip_on=False)

    if summ:  # summary column: mean ±1 std across words of per-word token-means
        sax = fig.add_axes([(MARGIN["l"] + iw + 14) / tw, MARGIN["b"] / h,
                            36 / tw, ih / h], sharey=ax)
        style_axis(sax, pal, lo, hi)
        sax.tick_params(axis="y", labelleft=False, length=0)
        sax.set_xlim(-0.5, 0.5)
        sax.set_xticks([])
        sax.set_xlabel("word\nmeans", fontsize=10 * PT, color=pal["ink2"])
        for i, c in enumerate(summ):
            xi = (i - (len(summ) - 1) / 2) * (5 / 36)
            sax.errorbar(xi, c["sm"]["mean"], yerr=c["sm"]["std"],
                         color=pal[c["id"]], linewidth=2 * PT, capsize=3,
                         capthick=2 * PT)
            sax.plot(xi, c["sm"]["mean"], linestyle="none",
                     marker=MARKERS[c["id"]], color=pal[c["id"]],
                     markersize=4.5, clip_on=False)

    suffix = " − no mention" if cfg.delta else ""
    legend = [(f"{c['label']}{suffix} — n={c['n']}", pal[c["id"]],
               MARKERS[c["id"]], "-") for c in conds if c["vals"] is not None]
    finish(fig, legend, cfg, pal, out, tw, h,
           "line: mean over words" + (" · shaded: ±1 std" if cfg.bands else ""))


# ---- layers chart (docs/layers.html) ---------------------------------------

def layers_series(data: dict, cfg: Config) -> list[dict]:
    vm = data[cfg.meas]["complete" if cfg.complete else "all"]
    family = vm["deltas"] if cfg.delta else vm["conds"]
    conds = CONDS_DELTA if cfg.delta else COND
    return [{"id": cid, "label": label, **family[cid]}
            for cid, label in conds if cid in family]


def draw_layers(conds: list[dict], n_layers: int, cfg: Config, pal: dict,
                out: Path) -> None:
    # the page's 400px frame: 16px top margin, 48px bottom, plus legend row
    top = MARGIN["t"] + 26 + (24 if cfg.title else 0)
    bot = 48
    ih = 400 - 16 - bot
    h = top + ih + bot
    iw = W - MARGIN["l"] - MARGIN["r"]
    tw = W

    fig = plt.figure(figsize=(tw / 96, h / 96), dpi=cfg.dpi)
    fig.patch.set_facecolor("none" if cfg.transparent else pal["surface"])
    ax = fig.add_axes([MARGIN["l"] / tw, bot / h, iw / tw, ih / h])

    lo = hi = 0.0
    for c in conds:
        for v, s in zip(c["mean"], c["std"]):
            lo = min(lo, v - (s if cfg.bands else 0))
            hi = max(hi, v + (s if cfg.bands else 0))
    lo, hi = pad_range(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, n_layers - 1)
    style_axis(ax, pal, lo, hi)
    fmt_ticks(ax, lo, hi, pal)
    ax.set_xticks(range(0, n_layers, 10), [str(i) for i in range(0, n_layers, 10)],
                  fontsize=11 * PT, color=pal["ink2"])
    ax.set_xlabel("layer", fontsize=11.5 * PT, color=pal["ink2"])
    dp = "Δ vs no mention: " if cfg.delta else ""
    ax.set_ylabel(dp + "mean cosine with concept vector",
                  fontsize=11.5 * PT, color=pal["ink2"], labelpad=2)

    x = np.arange(n_layers)
    for c in conds:
        if cfg.bands:
            m, s = np.asarray(c["mean"]), np.asarray(c["std"])
            ax.fill_between(x, m - s, m + s, color=pal[c["id"]], alpha=0.18,
                            linewidth=0)
    for c in conds:
        ax.plot(x, c["mean"], color=pal[c["id"]], linewidth=2 * PT,
                solid_capstyle="round", solid_joinstyle="round")
        ax.plot(x, c["mean"], linestyle="none", marker=MARKERS[c["id"]],
                color=pal[c["id"]], markersize=3, clip_on=False)

    suffix = " − no mention" if cfg.delta else ""
    legend = [(f"{c['label']}{suffix} — n={c['n']}", pal[c["id"]],
               MARKERS[c["id"]], "-") for c in conds]
    finish(fig, legend, cfg, pal, out, tw, h,
           "line: mean over words" + (" · shaded: ±1 std" if cfg.bands else ""))


# ---- forest chart (docs/forest.html) ---------------------------------------

def forest_rows(data: dict, cfg: Config, layer: int) -> list[dict]:
    vm = data[cfg.meas]["complete" if cfg.complete else "all"]
    rows = []
    for wi, word in enumerate(data["words"]):
        r = {"word": word}
        for cid, _ in CONDS_DELTA:
            b = vm[cid]
            r[cid] = ({"v": b["mean"][wi][layer], "sd": b["std"][wi][layer],
                       "n": b["n"][wi]} if b["mean"][wi] else None)
        if r["think"] or r["dont_think"]:
            rows.append(r)
    if cfg.sort == "alpha":
        rows.sort(key=lambda r: r["word"])
    else:
        key = "dont_think" if cfg.sort == "dont" else "think"
        rows.sort(key=lambda r: r[key]["v"] if r[key] else -1e9, reverse=True)
    return rows


def draw_forest(rows: list[dict], cfg: Config, pal: dict, out: Path) -> None:
    row_h = 16
    top = 10 + 26 + (24 if cfg.title else 0)
    bot = 42
    ih = len(rows) * row_h
    h = top + ih + bot
    ml = 118
    iw = W - ml - MARGIN["r"]
    tw = W

    fig = plt.figure(figsize=(tw / 96, h / 96), dpi=cfg.dpi)
    fig.patch.set_facecolor("none" if cfg.transparent else pal["surface"])
    ax = fig.add_axes([ml / tw, bot / h, iw / tw, ih / h])
    ax.set_facecolor("none")
    for side in ax.spines.values():
        side.set_visible(False)

    lo = hi = 0.0
    for r in rows:
        for cid, _ in CONDS_DELTA:
            d = r[cid]
            if d:
                lo, hi = min(lo, d["v"] - d["sd"]), max(hi, d["v"] + d["sd"])
    lo, hi = pad_range(lo, hi, 0.05)
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(rows) - 0.5, -0.5)  # first row on top, like the page

    ticks = np.linspace(lo, hi, 7)
    fmt = (lambda v: f"{v:.3f}") if abs(hi - lo) < 0.05 else (lambda v: f"{v:.2f}")
    ax.set_xticks(ticks, [fmt(v) for v in ticks], fontsize=11 * PT, color=pal["ink2"])
    ax.set_yticks(range(len(rows)), [r["word"] for r in rows],
                  fontsize=11 * PT, fontfamily=MONO, color=pal["ink"])
    ax.grid(axis="x", color=pal["line"], linewidth=0.75)
    ax.set_axisbelow(True)
    if lo < 0 < hi:
        ax.axvline(0, color=pal["ink2"], linewidth=0.75)
    ax.tick_params(length=0)
    ax.set_xlabel("Δ cosine with concept vector vs no mention",
                  fontsize=11.5 * PT, color=pal["ink2"])

    off = {"think": -2.5 / row_h, "dont_think": 2.5 / row_h}
    for i, r in enumerate(rows):
        for cid, _ in CONDS_DELTA:
            d = r[cid]
            if not d:
                continue
            y = i + off[cid]
            ax.plot([d["v"] - d["sd"], d["v"] + d["sd"]], [y, y],
                    color=pal[cid], linewidth=1.5 * PT, alpha=0.8,
                    solid_capstyle="butt")
            ax.plot(d["v"], y, linestyle="none", marker=MARKERS[cid],
                    color=pal[cid], markersize=4.5, clip_on=False)

    legend = [(f"{label} − no mention", pal[cid], MARKERS[cid], "-")
              for cid, label in CONDS_DELTA]
    finish(fig, legend, cfg, pal, out, tw, h,
           "dot: mean over sentences · whisker: ±1 std")


def meta_line(index: dict, cfg: Config, word: str, sentence: str, layer: int) -> str:
    """The viewer's footer line, for use as a slide caption."""
    if cfg.meas == "nla":
        j = index.get("nla_judge") or {}
        judge = f"{j['model']} (prompt {j['prompt_version']})" if j else "LLM"
        detail = (f"NLA: kitft/nla-gemma3-27b-L41-av (greedy) on layer {layer} resid_post — "
                  f"{judge} judge, 0–100 logit-expectation score for “{word.lower()}”")
    elif cfg.meas in ("sae", "sae_v2"):
        agg = "average" if cfg.agg == "mean" else cfg.agg
        sel = ("selection v2: contrastive vs 99 baseline words, excluded on 100 "
               "control-word prompts" if cfg.meas == "sae_v2"
               else "selection v1: raw concept-token activation, excluded on the "
                    "50 experiment sentences")
        detail = (f"SAE: Gemma Scope 2 residual 16k l0_medium, layer {layer}, "
                  f"{agg} over selected latents · {sel}")
    else:
        method = ("paper method (last prompt token of “Tell me about {word}.” − "
                  "99-word baseline mean)" if cfg.meas == "paper"
                  else "word-token method (mean over the word’s own tokens across "
                       "4 templates − baseline mean)")
        detail = f"concept vectors: {method}, layer {layer} of {index['n_layers'] - 1}"
    return (f"Run {index['run_id']} · google/gemma-3-27b-it (bf16, greedy) · "
            f"sentence: “{sentence}” · {detail}")


def meas_short(meas: str) -> str:
    return "paper method" if meas == "paper" else "word-token method"


def require_cv(cfg: Config) -> None:
    if cfg.meas not in ("word_tokens", "paper"):
        raise SystemExit(f"the {cfg.chart} chart only has the concept-vector "
                         "variants (word_tokens, paper)")


def run_prefix(index: dict) -> str:
    return f"Run {index['run_id']} · google/gemma-3-27b-it (bf16, greedy)"


def load_agg(name: str) -> dict:
    p = DOCS_DATA / "agg" / name
    if not p.exists():
        raise SystemExit(f"{p} missing — run scripts/export_agg_data.py")
    return load_gz(p)


def suffixes(cfg: Config, delta_applies: bool = True) -> str:
    s = "_delta" if cfg.delta and delta_applies else ""
    return s + ("" if cfg.complete else "_all") + f"_{cfg.theme}"


def main_aggregate(cfg: Config, index: dict) -> None:
    if cfg.meas == "sae_v2" and "v2" not in index.get("sae_versions", ["v1"]):
        raise SystemExit("this export has no sae_v2 series — re-run export_viz_data.py")
    if cfg.meas == "nla":
        layer = index["nla_layer"]
    elif cfg.meas in ("sae", "sae_v2"):
        layer = min(index["sae_layers"], key=lambda l: abs(l - cfg.layer))
    else:
        layer = max(0, min(index["n_layers"] - 1, cfg.layer))
    chunk = load_agg(f"{cfg.sent}.json.gz")
    conds = agg_series(chunk, cfg, layer, index["sae_layers"])
    if not any(c["vals"] is not None for c in conds):
        raise SystemExit("no plottable series for this selection")
    out = cfg.out or (ARTIFACTS / "figures" /
                      f"agg_s{cfg.sent:02d}_{cfg.meas}_L{layer}{suffixes(cfg)}.png")
    draw_aggregate(chunk["tokens"], conds, cfg, THEMES[cfg.theme], out)
    mode = ("complete cases (all three conditions exact)" if cfg.complete
            else "all words with an exact completion per condition")
    stat = ("paired per-word Δ vs no mention, mean ±1 std across words"
            if cfg.delta else "mean ±1 std across concept words")
    if cfg.meas == "nla":
        detail = f"NLA judge score (0–100) on layer {layer} resid_post"
    elif cfg.meas in ("sae", "sae_v2"):
        agg = "average" if cfg.agg == "mean" else cfg.agg
        detail = (f"SAE: Gemma Scope 2 residual 16k l0_medium, layer {layer}, "
                  f"{agg} over selected latents "
                  f"(selection {'v2, contrastive' if cfg.meas == 'sae_v2' else 'v1'})")
    else:
        detail = (f"concept vectors: {meas_short(cfg.meas)}, "
                  f"layer {layer} of {index['n_layers'] - 1}")
    print(f"{run_prefix(index)} · sentence: “{chunk['sentence']}” · "
          f"{stat}, {mode} · {detail}")
    print(f"wrote {out}")


def main_layers(cfg: Config, index: dict) -> None:
    require_cv(cfg)
    conds = layers_series(load_agg("layers.json.gz"), cfg)
    if not conds:
        raise SystemExit("no plottable series for this selection")
    out = cfg.out or (ARTIFACTS / "figures" /
                      f"layers_{cfg.meas}{suffixes(cfg)}.png")
    draw_layers(conds, index["n_layers"], cfg, THEMES[cfg.theme], out)
    mode = ("complete cases (all three conditions exact per sentence)"
            if cfg.complete else "all words with an exact completion per condition")
    stat = ("paired per-word Δ vs no mention, mean ±1 std across words"
            if cfg.delta else "mean ±1 std across concept words")
    print(f"{run_prefix(index)} · token- and sentence-collapsed, {stat}, {mode} · "
          f"concept vectors: {meas_short(cfg.meas)}, layers 0–{index['n_layers'] - 1}")
    print(f"wrote {out}")


def main_forest(cfg: Config, index: dict) -> None:
    require_cv(cfg)
    layer = max(0, min(index["n_layers"] - 1, cfg.layer))
    rows = forest_rows(load_agg("words.json.gz"), cfg, layer)
    if not rows:
        raise SystemExit("no plottable rows for this selection")
    out = cfg.out or (ARTIFACTS / "figures" /
                      f"forest_{cfg.meas}_L{layer}{suffixes(cfg, delta_applies=False)}.png")
    draw_forest(rows, cfg, THEMES[cfg.theme], out)
    mode = ("complete cases (all three conditions exact per sentence)"
            if cfg.complete else "all sentences with exact completions for the pair")
    print(f"{run_prefix(index)} · layer {layer} of {index['n_layers'] - 1} · "
          f"paired per-word Δ vs no mention, token-mean per sentence · {mode} · "
          f"concept vectors: {meas_short(cfg.meas)}")
    print(f"wrote {out}")


def main(cfg: Config) -> None:
    index = json.loads((DOCS_DATA / "index.json").read_text())
    if cfg.chart == "aggregate":
        return main_aggregate(cfg, index)
    if cfg.chart == "layers":
        return main_layers(cfg, index)
    if cfg.chart == "forest":
        return main_forest(cfg, index)
    if cfg.word is None:
        print(f"run {index['run_id']}: {len(index['words'])} words, "
              f"sentences {index['sentence_order'][0]}–{index['sentence_order'][-1]}")
        print(", ".join(index["words"]))
        return
    by_lower = {w.lower(): w for w in index["words"]}
    word = by_lower.get(cfg.word.lower())
    if word is None:
        raise SystemExit(f"unknown word {cfg.word!r} — run without --word to list")
    if cfg.meas == "sae_v2" and "v2" not in index.get("sae_versions", ["v1"]):
        raise SystemExit("this export has no sae_v2 series — re-run export_viz_data.py")
    if cfg.meas in ("sae", "sae_v2", "nla") and cfg.base == "band":
        cfg.base = "none"  # like the viewer: bands only exist for concept vectors

    if cfg.meas == "nla":
        layer = index["nla_layer"]
    elif cfg.meas in ("sae", "sae_v2"):
        layer = min(index["sae_layers"], key=lambda l: abs(l - cfg.layer))
    else:
        layer = max(0, min(index["n_layers"] - 1, cfg.layer))

    si = str(cfg.sent)
    slot = load_slot(index, word, si)
    conds = series_data(slot, cfg, layer, index["sae_layers"])
    for c in conds:
        if c["excluded"]:
            print(f"{c['label']}: excluded — model wrote “{c['completion']}”")
    if not any(c["vals"] is not None for c in conds):
        raise SystemExit("no plottable series for this selection "
                         "(all conditions excluded or measurement missing)")

    out = cfg.out or (ARTIFACTS / "figures" /
                      f"{word}_s{cfg.sent:02d}_{cfg.meas}_L{layer}_{cfg.theme}.png")
    draw(slot["tokens"], conds, cfg, THEMES[cfg.theme], out)

    print(meta_line(index, cfg, word, slot["sentence"], layer))
    if cfg.meas in ("sae", "sae_v2"):
        meta_key = "sae_latents_v2" if cfg.meas == "sae_v2" else "sae_latents"
        for m in slot.get(meta_key, []):
            if m["layer"] == layer:
                for e in m["latents"]:
                    print(f"  #{e['latent']}  {e['label'] or '(no label)'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
