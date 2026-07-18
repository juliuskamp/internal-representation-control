# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Replication of the "intentional control" experiment from the paper *Emergent Introspective Awareness in Large Language Models* (excerpts in `scratch/`), on `google/gemma-3-27b-it` with Gemma Scope 2 SAEs. The model is told to write a fixed sentence while thinking / not thinking about a concept word; we measure how strongly the concept is internally represented on the response tokens via (a) cosine with mean-difference concept vectors and (b) activation of concept-selective SAE latents. Findings and decisions are logged in `notes/` (`smoke_tests.md`, `run1-core_results.md`) — read these before changing measurement code.

## Commands

Everything runs through `uv` (no test suite or linter is configured):

```bash
# Full pipeline. A "run" is an incrementally grown dataset: re-invoking with
# the same --run-id extends it (stages are cached/resumable; safe to re-run).
uv run python scripts/run_pipeline.py --run-id run1-core
uv run python scripts/run_pipeline.py --run-id run1-core --stages generate measure
uv run python scripts/run_pipeline.py --run-id run1-core --words Dust Oceans --sentences-per-word 5

# Interactive viewers for a finished run (static image export of a viewer
# chart: scripts/render_viewer_figure.py)
uv run python scripts/export_viz_data.py --run-id run1-core  # writes docs/data/
uv run python scripts/export_agg_data.py   # docs/data/agg/ for aggregate.html
python -m http.server -d docs                                # view at localhost:8000

# Smoke tests (a: generation, b: SAE, c: concept vector, d: latent selection)
uv run python scripts/smoke_a_generate.py

# NLA explanations (decode stored activations to text; see notes/nla_setup.md)
bash nla_server/setup.sh && bash nla_server/launch.sh   # SGLang server, own py3.12 venv
uv run python scripts/nla_explain.py --run-id run1-core --words Dust --limit 3
```

Requires a GPU with ~55 GB VRAM (bf16) and `.env` at the repo root with `HF_TOKEN` (see `.env.example`). The `measure` stage and `export_viz_data.py` don't need the LLM, only stored activations + SAEs.

## Hard requirements

- **`from irc import env` must be the first import in every entry point** — it loads `.env` and sets `HF_HOME=/workspace/hf-cache` before `huggingface_hub` reads it at import time.
- **bf16 only, never quantized/fp32** — we measure activations; dtype changes them.
- **Exclude token position 0 (BOS) from all activation measurements.** Its residual norm is ~20× other tokens and the SAEs were not trained on it (see `notes/smoke_tests.md`).
- **SAE variant is pinned to `gemma-scope-2-27b-it-res` / `layer_{n}_width_16k_l0_medium`** — the only variant Neuronpedia indexed for this model (`{layer}-gemmascope-2-res-16k`); other L0 variants have non-matching latent indices, breaking label lookups.
- **Word lists are versioned, never edited in place**: `irc/words_paper.py` is transcribed from the paper (do not edit; baseline deduplicated to 99 words per logged decision), `irc/words.py` sets are `_V1` — add a new version if a set must change.
- Concept words are stored Capitalized but must always be lowercased when placed into prompts.

## Architecture

`irc/` package, orchestrated by `scripts/run_pipeline.py` in four cached stages (`irc/pipeline.py`):

1. **vectors** — mean-difference concept vectors for 50 concept + 100 control words, two extraction variants: `"paper"` (last-prompt-token of "Tell me about {word}.") and `"word_tokens"` (mean over the word's own token positions across 4 templates, `irc/concept_vectors.py`). Cached in `artifacts/concept_vectors/bank_{variant}_v1.pt`. Note: the paper's extraction positions failed sanity checks on Gemma (they encode chat-template structure); `word_tokens` is the working method, and raw cosines are dominated by a shared generic direction — paired within-word comparisons (or centering) are the sensitive test.
2. **generate** — per (word, sentence) × condition (prompts in `irc/conditions.py`; `no_mention` is our word-free baseline, shared across words per sentence): greedy generation, **exact-output compliance check** (non-exact completions are flagged and excluded from measurement), all-layer resid_post capture on response tokens via `ResidualCapture` hooks (`irc/model.py`). Written to `artifacts/runs/{run_id}/` as `generations.jsonl` + `acts/*.pt` (bf16, layers × tokens × d_model).
3. **latents** — per concept word and SAE layer (16/31/40/53): select top-k latents with high mean activation on the word's tokens in templates and near-zero max activation on the 50 experiment sentences (`baseline_max < 0.1 × concept_mean`), cross-checked with Neuronpedia auto-interp labels (cached in `artifacts/neuronpedia_cache.json`). Output: `artifacts/latents_v1/{word}.json`. Word-independent of run_id.
4. **measure** — model-free; reads stored acts. Concept-vector cosines per layer×token (target word + 100-control-word null) → `results/concept_cosines.parquet`, `token_cosines/`, `null_means/`; selected-latent SAE stats → `results/sae_latents.parquet`.

Run provenance: every invocation (config, versions, git commit) is appended to `artifacts/runs/{run_id}/invocations.jsonl`; `config.json` is a snapshot of the latest invocation only — the run's data may be the union of many invocations.

**Viewer**: `docs/` is a static site for GitHub Pages (also embeddable via iframe; `?embed=1` hides the page chrome). `docs/index.html` fetches chunked data from `docs/data/` — `index.json` (metadata + word list), `shared-bands.json.gz` (word-independent `no_mention` null bands, deduplicated), `words/{word}.json.gz` (per-word slots, lazy-loaded). `scripts/export_viz_data.py` writes these from a run's stored activations; the data files are committed derived data, so publishing a new run means re-export + commit. fetch() is blocked on `file://` — always view through an HTTP server. `docs/aggregate.html` is a second page showing mean ±1 std across words per sentence; it reads `docs/data/agg/{si}.json.gz`, written by `scripts/export_agg_data.py` *from the word chunks* (re-run it after every export_viz_data.py run). `docs/layers.html` is a third page: concept-vector strength vs layer, collapsed over tokens+sentences (words as replicates); it reads `docs/data/agg/layers.json.gz` from the same export script.

`artifacts/`, `scratch/`, and `.env` are gitignored — artifacts are the (large) data store, not code. `docs/data/` is deliberately tracked (it is the published site).
