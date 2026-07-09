"""Pinned experiment constants — single source of truth.

These values are experiment-defining: changing any of them makes new
measurements incomparable with previously generated artifacts (see CLAUDE.md).
Import from here instead of re-declaring in scripts.
"""

# SAE variant is pinned to the only release Neuronpedia indexed for
# gemma-3-27b-it ("{layer}-gemmascope-2-res-16k"); other L0 variants have
# non-matching latent indices, breaking label lookups.
SAE_RELEASE = "gemma-scope-2-27b-it-res"
SAE_ID_TEMPLATE = "layer_{layer}_width_16k_l0_medium"

# SAE layers measured throughout (the Gemma Scope 2 release layers we use).
SAE_LAYERS: tuple[int, ...] = (16, 31, 40, 53)

# Concept-vector extraction variants computed by the vectors stage
# (see irc/concept_vectors.py; "word_tokens" is the working method).
VECTOR_VARIANTS: tuple[str, ...] = ("paper", "word_tokens")

# Decoder depth of gemma-3-27b-it: layer count of all-layer residual captures.
N_LAYERS = 62

# NLA actor (activation verbalizer), trained on resid_post of decoder block 41
# (extraction_layer_index in the checkpoint's nla_meta.yaml; other layers are
# out-of-distribution for it). See notes/nla_setup.md.
NLA_REPO = "kitft/nla-gemma3-27b-L41-av"
NLA_LAYER = 41
