# SAE latent selection v2 (2026-07-10)

New versioned latent set `artifacts/latents_v2/` alongside v1 — v1 files and all
existing results are untouched. Selected with
`scripts/run_pipeline.py --stages latents --latents-version v2`; measured into
`results/sae_latents_v2.parquet` (v1's `sae_latents.parquet` keeps its name).

## Why v1 needed a successor

Compared against SAEBench-style sparse-probing practice, v1 had two flaws:

1. **No contrast.** v1 ranked latents by raw mean activation on the concept
   word's token span across the 4 templates. Generic latents ("noun being
   discussed", template phrasing) score high for *every* word and can crack the
   top-5 — visible in v1's Neuronpedia labels (e.g. Dust L31: "researching
   diverse topics", "asking how and about aspects").
2. **Circular exclusion.** v1's eligibility filter (`baseline_max <
   0.1 × concept_mean`) computed `baseline_max` over the same 50 experiment
   sentences later used for measurement, so `no_mention ≈ 0` held by
   construction rather than being a finding.

## What changed (three substitutions; everything else identical to v1)

Same SAE variant/layers, top-k=5, word-token-span statistic, 4 templates,
`< 0.1 ×` threshold form, Neuronpedia cross-check.

1. **More words through the templates.** Per-latent template statistics are
   computed for the 99 `BASELINE_WORDS_PAPER` and 100 `CONTROL_WORDS_PAPER` in
   addition to the 50 concept words. Cached per word in
   `artifacts/latents_v2/_word_template_means.pt` and
   `_control_max.pt` (analogous to v1's `_baseline_max.pt`).
2. **Contrastive score.** Rank = `concept_mean(word) − mean over the 99
   BASELINE words of the same statistic` (mean over the word's own token span,
   averaged over the 4 templates — apples-to-apples with v1's `concept_mean`).
   Top-5 of eligible latents, entries with score ≤ 0 dropped (so a word/layer
   may keep fewer than 5).
3. **Exclusion on control words, not test sentences.** `baseline_max` = max
   activation over **all token positions (BOS excluded)** of the 100 CONTROL
   words' template prompts — all-token on purpose so template-phrasing latents
   get excluded too. Eligibility stays `baseline_max < 0.1 × concept_mean`
   (raw concept_mean as the reference, keeping v1's threshold semantics). The
   50 experiment sentences are never touched during selection.

Per-latent JSON fields: v1's `concept_mean`/`baseline_max`/`label` plus
`contrast_score` (the ranking score) and `baseline_word_mean` (the subtracted
mean; `contrast_score = concept_mean − baseline_word_mean`). `baseline_max`
now means the control-word all-token max, *not* the v1 experiment-sentence max.

**Comparability warning:** because the `no_mention` floor is no longer ≈ 0 by
construction, v2 measurement numbers are not directly comparable to the v1
tables in `notes/run1-core_results.md`. A `no_mention` floor above ~0 under v2
is an expected, honest empirical outcome, not a bug.

## Selection diagnostics (v1 vs v2, all 50 words)

Top-5 overlap per layer (mean |v1 ∩ v2| out of 5; distribution over words):

| layer | mean overlap | words with 0 common |
|------:|-------------:|--------------------:|
| 16 | 1.80 | 6 |
| 31 | 0.76 | 23 |
| 40 | 1.66 | 9 |
| 53 | 0.90 | 14 |

v2 keeps fewer than 5 latents for many word/layers (score ≤ 0 dropped, stricter
exclusion): mean kept = 4.4 (L16), 3.0 (L31), 3.9 (L40), 3.8 (L53). Words with
**zero** kept latents at some layer: Bags/Snow/Trees (L31), Deserts/Oceans/
Snow/Vegetables (L40), Bags/Trees (L53) — their v2 SAE measurement rows are
simply absent at those layers.

Label quality: the generic v1 latents are gone as intended — e.g. Dust L31 v1
had "researching diverse topics" / "asking how and about aspects" / "code
comments and specific words"-type latents; v2 keeps "air pollution and health
effects". Amphitheaters gains "stadium spectators cheering", Blood "ancestry
and lineage".

**Trade-off found:** the control-word exclusion also removes some genuinely
concept-selective latents when a control word is semantically adjacent.
Aquariums L40 latent 9322 ("aquarium and tank", concept_mean 3354, contrast
3354) is excluded because control_max = 761 ≫ 335 (0.1 × concept_mean) — some
control word's template prompt legitimately activates it. Same mechanism for
Bags ("clothing descriptions") and the zero-latent words above (Snow, Trees,
Oceans, Deserts, Vegetables all have semantically close control words). v2 is
strict/conservative by construction: cleaner floors at the cost of recall for
words whose concept neighborhoods overlap the control list.

## Results on run1-core (full 50×50 grid, 2500 pairs)

Selected-latent summed activation on response tokens (`act_sum_mean`, mean over
pairs), v1 numbers recomputed from `results/sae_latents.parquet` on the same
grid (the table in `run1-core_results.md` was the earlier 2-sentence run):

| layer | v1 think | v1 dont | v1 no_mention | | v2 think | v2 dont | v2 no_mention |
|------:|---------:|--------:|--------------:|-|---------:|--------:|--------------:|
| 16 | 1.6 | 1.5 | 0.07 | | 1.7 | 1.6 | 0.83 |
| 31 | 57.5 | 1.8 | 0.24 | | 40.8 | 1.3 | 0.58 |
| 40 | 267.4 | 6.8 | 0.21 | | 130.7 | 7.0 | 3.21 |
| 53 | 437.3 | 10.9 | 1.85 | | 157.3 | 7.1 | 5.60 |

Any-latent-active fraction of runs (%):

| layer | v1 think | v1 dont | v1 no_m | | v2 think | v2 dont | v2 no_m |
|------:|---------:|--------:|--------:|-|---------:|--------:|--------:|
| 16 | 13.1 | 12.1 | 1.6 | | 15.2 | 14.3 | 6.8 |
| 31 | 42.8 | 4.0 | 0.8 | | 30.1 | 2.7 | 1.4 |
| 40 | 56.1 | 3.5 | 0.3 | | 39.6 | 4.6 | 2.3 |
| 53 | 58.8 | 4.3 | 1.3 | | 28.2 | 3.1 | 2.2 |

Reading:
- **The no_mention floor rose above ~0, as expected** — it is now an empirical
  result (selection never saw the experiment sentences), not true by
  construction. It stays small: think/no_mention ratio ≈ 70× (L31), 41× (L40),
  28× (L53).
- **The core finding survives the fairer selection.** think ≫ dont_think ≈
  no_mention at deep layers; "don't think" still suppresses near-completely
  (dont_think within ~2× of the empirical floor at every layer).
- Absolute v2 magnitudes are smaller than v1 — fewer latents per word (drop
  ≤ 0 + stricter exclusion) and the removal of high-magnitude generic latents.
  **v1 and v2 numbers are not directly comparable**, between versions or with
  the tables in `run1-core_results.md`.
- L16 remains lexical/weak in both versions; its v2 floor (0.83) is the noisiest.

## Files

- `artifacts/latents_v2/{word}.json`, `_word_template_means.pt`,
  `_control_max.pt` — selection outputs + caches.
- `artifacts/runs/run1-core/results/sae_latents_v2.parquet` — 25,941 rows
  (fewer than v1's 27,136 because empty word/layer selections produce no rows).
- Provenance: `latents_version` recorded in `invocations.jsonl` / `config.json`.
- Viewer/export (`scripts/export_viz_data.py`, `docs/`) still read v1 — not
  repointed.
