# Smoke test results (2026-07-08)

Environment: RTX PRO 6000 Blackwell (96 GB), `/workspace` 200 GB persistent NVMe,
`HF_HOME=/workspace/hf-cache`. uv-managed env, torch 2.12.1+cu130, transformers 5.13,
sae-lens 6.45.3. Model: google/gemma-3-27b-it, bf16 (`Gemma3ForConditionalGeneration`),
54.9 GB VRAM, 62 decoder layers, d_model 5376.

## A — model load + instruction following: PASS
All three prompt variants (think / don't-think / baseline) greedily produce exactly
`He lives in the red house.` and nothing else. Peak VRAM 54.9 GB.

## B — Gemma Scope 2 SAE: PASS (with BOS caveat)
`gemma-scope-2-27b-it-res / layer_31_width_16k_l0_medium` (d_in 5376 → 16384,
hook `blocks.31.hook_resid_post`, matches our capture point = decoder layer output).
Per-token relative reconstruction error 0.04–0.14 on content tokens.

**Trap found: BOS (position 0) residual norm is ~800k vs ~40k for other tokens; the SAE
was not trained on it (rel. err 2.36, L0 ≈ 6700 there). Exclude position 0 everywhere.**

## C — concept vector: PASS with a method change
The two originally sketched extraction positions both FAILED sanity checks:
- `last_prompt` (final prompt token of "Tell me about X") and `response_mean`
  (mean over generated response) produce vectors dominated by chat-template/position
  structure — probe cosines identical for football vs unrelated text (sep ≈ 0.01).

Working method — **word_tokens**: mean resid activation at the concept word's own
tokens across 4 sentence templates (WORD_TEMPLATES_V1), minus the mean over
RANDOM_WORDS_V1 (30 words). Evaluation must **center** probe activations by a
generic-corpus mean before cosine (raw cosines are dominated by a shared generic
direction).

Centered-cosine sanity (football vector):
- token "▁football" in held-out text: +0.76…+0.90 (layers 10–55)
- football-context text *without* the word (goalkeeper/penalty): text-mean
  +0.13/+0.15/+0.10 at L22/31/40 → contextual generalization
- unrelated texts: ≈ 0 (−0.11…+0.07)
- the template's trailing "\n" token shows a constant high cosine in ALL conditions
  (high-norm template direction) → measure only sentence content tokens.

## D — SAE latent selection + Neuronpedia: PASS
Data-driven selection (max-activation on concept texts minus baseline texts),
layer 31, l0_medium: top latents 7106 "football pass", 8021 "sports and athletic
activity", 2860 "sports events and equipment", 1073 "game outcomes and strategies",
133 "athleticism, championship, captain" — 5/8 concept-related. The 3 junk latents
all had high baseline activity → **add a baseline_mean ≈ 0 filter before ranking.**

Neuronpedia mapping (confirmed by API): model `gemma-3-27b-it`, source
`{layer}-gemmascope-2-res-16k` = SAELens release `gemma-scope-2-27b-it-res`,
sae_id `layer_{n}_width_16k_l0_medium`. Only this L0 variant is indexed — use
l0_medium wherever Neuronpedia labels matter. Feature endpoint:
`GET /api/feature/gemma-3-27b-it/{layer}-gemmascope-2-res-16k/{index}` →
`explanations[0].description`.

## Decisions for the full pipeline
- Concept vectors: word_tokens extraction; centered projection/cosine as the measure;
  exclude BOS and all chat-template/special tokens; measure on sentence content tokens.
- SAEs: `gemma-scope-2-27b-it-res`, 16k, l0_medium (Neuronpedia-aligned).
- Latent selection: require near-zero baseline activation, then rank by concept activation.
