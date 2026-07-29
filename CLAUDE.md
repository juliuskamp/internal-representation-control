# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Replication of the "intentional control" experiment from the paper *Emergent Introspective Awareness in Large Language Models* (excerpts in the gitignored `scratch/`), on `google/gemma-3-27b-it` with Gemma Scope 2 SAEs. The model is told to write a fixed sentence while thinking / not thinking about a concept word; we measure how strongly the concept is internally represented on the response tokens via (a) cosine with mean-difference concept vectors, (b) activation of concept-selective SAE latents, and (c) NLA decodes of the activations scored by an LLM judge.

## Commands

Everything runs through `uv` (Python 3.13; no test suite or linter is configured):

```bash
# Full pipeline. A "run" is an incrementally grown dataset: re-invoking with
# the same --run-id extends it (stages are cached/resumable; safe to re-run).
uv run python scripts/run_pipeline.py --run-id run1-core
uv run python scripts/run_pipeline.py --run-id run1-core --stages generate measure
uv run python scripts/run_pipeline.py --run-id run1-core --words Dust Oceans --sentences-per-word 5

# Interactive viewers for a finished run (static image export of any of the
# four viewer charts: scripts/render_viewer_figure.py --chart word|aggregate|layers|forest,
# or by passing a viewer URL as the positional argument)
uv run python scripts/export_viz_data.py --run-id run1-core  # writes docs/data/
uv run python scripts/export_agg_data.py   # docs/data/agg/, from the word chunks
python -m http.server -d docs                                # view at localhost:8000

# Smoke tests (a: generation, b: SAE, c: concept vector, d: latent selection)
uv run python scripts/smoke_a_generate.py

# NLA explanations (decode stored activations to text; see nla_server/README.md)
bash nla_server/setup.sh && bash nla_server/launch.sh   # SGLang server, own py3.12 venv
uv run python scripts/nla_explain.py --run-id run1-core --words Dust --agg token --limit 3
uv run python scripts/nla_judge.py --run-id run1-core   # needs OPENROUTER_API_KEY
```

Requires a GPU with ~55 GB VRAM (bf16) and `.env` at the repo root with `HF_TOKEN` (see `.env.example`). The `measure` stage is model-free (stored acts only); `export_viz_data.py` needs a GPU for the SAE encode but not the LLM; `export_agg_data.py` and `nla_judge.py` need neither.

## Hard requirements

- **`from irc import env` must be the first import in every entry point** — it loads `.env` (including `HF_HOME`, if set) before `huggingface_hub` reads it at import time.
- **bf16 only, never quantized/fp32** — we measure activations; dtype changes them.
- **Exclude token position 0 (BOS) from all activation measurements.** Its residual norm is ~20× other tokens and the SAEs were not trained on it.
- **SAE variant is pinned to `gemma-scope-2-27b-it-res` / `layer_{n}_width_16k_l0_medium`** — the only variant Neuronpedia indexed for this model (`{layer}-gemmascope-2-res-16k`); other L0 variants have non-matching latent indices, breaking label lookups.
- **Word lists are versioned, never edited in place**: `irc/words_paper.py` is transcribed from the paper (do not edit; baseline deduplicated to 99 words per logged decision), `irc/words.py` sets are `_V1` — add a new version if a set must change.
- Concept words are stored Capitalized but must always be lowercased when placed into prompts.
- Experiment-defining values live in `irc/constants.py` (SAE release/layers, `N_LAYERS`, vector variants, `LATENTS_VERSION`, NLA repo/layer) — import them, never re-declare.

## Architecture

`irc/` package, orchestrated by `scripts/run_pipeline.py` in four cached stages (`irc/pipeline.py`):

1. **vectors** — mean-difference concept vectors for 50 concept + 100 control words, two extraction variants: `"paper"` (last-prompt-token of "Tell me about {word}.") and `"word_tokens"` (mean over the word's own token positions across 4 templates, `irc/concept_vectors.py`). Cached in `artifacts/concept_vectors/bank_{variant}_v1.pt`. Note: the paper's extraction positions failed sanity checks on Gemma (they encode chat-template structure); `word_tokens` is the working method, and raw cosines are dominated by a shared generic direction — paired within-word comparisons (or centering) are the sensitive test.
2. **generate** — per (word, sentence) × condition (prompts in `irc/conditions.py`; `no_mention` is our word-free baseline, shared across words per sentence): greedy generation, **exact-output compliance check** (non-exact completions are flagged and excluded from measurement), all-layer resid_post capture on response tokens via `ResidualCapture` hooks (`irc/model.py`). Written to `artifacts/runs/{run_id}/` as `generations.jsonl` + `acts/*.pt` (bf16, layers × tokens × d_model).
3. **latents** — per concept word and SAE layer (16/31/40/53): select top-k concept-selective latents, cross-checked with Neuronpedia auto-interp labels (cached in `artifacts/neuronpedia_cache.json`). Output: `artifacts/latents_{version}/{word}.json`, word-independent of run_id. Two selection methods exist: `v2` (the published one, `constants.LATENTS_VERSION`) scores latents contrastively against the 99 baseline words and excludes on the 100 control words' template prompts; `v1` ranks by raw activation on the word's own tokens and excludes on the 50 experiment sentences. Only v2 is exported to the viewer.
4. **measure** — model-free; reads stored acts. Concept-vector cosines per layer×token (target word + 100-control-word null) → `results/concept_cosines.parquet`, `token_cosines/`, `null_means/`; selected-latent SAE stats → `results/sae_latents_v2.parquet` (v1 would write the unsuffixed `sae_latents.parquet`, so re-measuring under a different selection never clobbers earlier results).

Optional NLA path (not a pipeline stage): `scripts/nla_explain.py` decodes stored layer-41 activations to text against the SGLang server in `nla_server/`, and `scripts/nla_judge.py` scores those explanations for concept presence via OpenRouter. Both are resumable and write into the run's `results/`.

Run provenance: every invocation (config, versions, git commit) is appended to `artifacts/runs/{run_id}/invocations.jsonl`; `config.json` is a snapshot of the latest invocation only — the run's data may be the union of many invocations.

**Viewer**: `docs/` is a static site for GitHub Pages (also embeddable via iframe; `?embed=1` hides the page chrome and makes the page post its height to the parent). `docs/index.html` fetches chunked data from `docs/data/` — `index.json` (metadata + word list), `shared-bands.json.gz` (word-independent `no_mention` null bands, deduplicated), `words/{word}.json.gz` (per-word slots, lazy-loaded). `scripts/export_viz_data.py` writes these from a run's stored activations; the data files are committed derived data, so publishing a new run means re-export + commit. fetch() is blocked on `file://` — always view through an HTTP server. `docs/aggregate.html` is a second page showing mean ±1 std across words per sentence; it reads `docs/data/agg/{si}.json.gz`, written by `scripts/export_agg_data.py` *from the word chunks* (re-run it after every export_viz_data.py run). Further summary pages from the same export script: `docs/layers.html` (concept-vector strength vs layer, collapsed over tokens+sentences; the replicate is selectable — words, or word×sentence cells, which probably is the paper's Figure 26 order — as is the band, ±1 std or ±1 SEM; reads `agg/layers.json.gz`) and `docs/forest.html` (per-word paired Δ vs no_mention at one layer, slider; reads `agg/words.json.gz`).

Viewer state is mirrored into the URL on every page, and `scripts/render_viewer_figure.py` parses those same URLs — so a link and a static figure are interchangeable. The legacy `meas=sae_v2` key (from when both latent selections were published) aliases to `sae` in both places; keep that alias.

`artifacts/`, `scratch/`, and `.env` are gitignored — artifacts are the (large) data store, not code. `docs/data/` is deliberately tracked (it is the published site).
