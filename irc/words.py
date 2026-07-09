"""Fixed, versioned word sets for concept-vector extraction.

RANDOM_WORDS_V1 is the baseline set for the mean-difference concept vectors.
Never edit in place — add a new version if the set must change, and log the
version with every run.
"""

RANDOM_WORDS_V1: list[str] = [
    "chair",
    "cloud",
    "violin",
    "pepper",
    "harbor",
    "lantern",
    "sister",
    "meadow",
    "copper",
    "pillow",
    "engine",
    "island",
    "butter",
    "mirror",
    "tunnel",
    "sparrow",
    "blanket",
    "pencil",
    "garden",
    "thunder",
    "bottle",
    "ladder",
    "carpet",
    "window",
    "helmet",
    "candle",
    "bridge",
    "forest",
    "hammer",
    "ocean",
]

# Templates where the word sits mid-sentence in varied contexts; activations are
# taken at the word's own tokens ("word_tokens" extraction variant).
WORD_TEMPLATES_V1: list[str] = [
    "Tell me about {word}.",
    "I've been thinking about {word} a lot lately.",
    "The topic of {word} came up in conversation yesterday.",
    "She wrote an essay about {word} for class.",
]
