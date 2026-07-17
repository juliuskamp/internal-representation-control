# run1-core results (2026-07-08)

Config: `artifacts/runs/run1-core/config.json` — 50 paper concept words x 2 sentences
(seed 0), conditions think / dont_think / no_mention, greedy decoding, bf16.

## Compliance (exact-output check)
243 generations, **35 flagged non-exact** (33 think, 2 dont_think) and excluded →
65 complete (word, sentence) pairs.
- Think-condition failures are mostly the model *weaving the word into the sentence*
  (e.g. Dynasties: "...a fleeting glory mirroring the rise and fall of dynasties.").
- 2 dont_think failures are *avoidance rewrites* of sentence content related to the
  word (Plastic: "bubble wrap" → "cushioning air pockets"; Snow: "covered with
  snow" → "covered with white") — behavioral pink-elephant leakage.
- 6 of the 35 are apostrophe-typography mismatches (model wrote ’ for ');
  strictly excluded per protocol.

## Concept vectors (response sentence tokens, paired across conditions)
Layers 15–55 mean, n=65 pairs, Wilcoxon signed-rank:
- **think > dont_think: 100% of pairs**, mean gap +0.0053 (paper vectors) /
  +0.0076 (word_tokens), p ≈ 2.4e-12 (both variants).
- think > no_mention: same magnitude/significance.
- dont_think vs no_mention: +0.0002 (55%, p=0.16, paper) / +0.0004 (62%, **p=0.03**,
  word_tokens) — at most a weak pink-elephant residual, unlike the clear
  above-baseline "don't think" effect the paper reports for Claude models.
- Gap profile across layers (word_tokens): positive from ~L10, peak ~+0.06 at
  L18–22, decays by the final layers — consistent with the paper's "representation
  decays back to baseline by the final layer" for recent models.
- Paper-variant vectors show a *negative* think gap at L10–28 (their generic/
  template contamination reacts to the instruction), positive after L40.
- Absolute cosines sit well inside the control-word 5–95% band (the band spans
  ±0.9 — Gemma's shared generic direction gives every vector a large
  word-specific but condition-constant offset). The paired within-word comparison
  is the sensitive test; the paper-style band test is inconclusive on Gemma
  without centering.

## SAE latents (Gemma Scope 2 16k l0_medium; top-5 per word, near-zero on all 50 sentences by construction)
> 2026-07-10: selection v2 (contrastive, non-circular exclusion) exists alongside
> this — see `notes/latents_v2.md`. Numbers below are v1 and not comparable to v2.
Selected-latent summed activation on response tokens (mean over pairs):

| layer | think | dont_think | no_mention |
|------:|------:|-----------:|-----------:|
| 16 | 1.5 | 1.1 | 0.04 |
| 31 | 54.1 | 0.3 | 0.3 |
| 40 | 302.9 | 1.8 | 0.0 |
| 53 | 467.8 | 10.9 | 3.3 |

Any-latent-active fraction of runs in think: 39% (L31), 54% (L40), 60% (L53) vs
~1–4% in the other conditions. Massive, clean separation: the model demonstrably
activates word-specific features while writing the unrelated sentence, and
suppresses them near-completely under "don't think".

## Caveats
- Excluded think-pairs (non-compliant generations) may be the strongest
  "thinkers" — exclusion is per-protocol but selective.
- SAE latent selection is word-token based (lexical); deep-layer results (40, 53)
  are the more concept-like evidence.
- dont_think ≈ no_mention on concept vectors is a *difference from the paper's
  Claude results* worth highlighting, not a bug.

## Next steps (not run)
- Incentive prompt variants (rewarded/punished/happy/sad/charity/terrorist) via
  `--conditions`.
- More sentences per word (`--sentences-per-word`), pairs extend deterministically.
- Centered-vector robustness measurement (subtract mean over all 150 vectors per
  layer) to shrink the control-word band.
