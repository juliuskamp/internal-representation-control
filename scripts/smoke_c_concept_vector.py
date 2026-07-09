"""Smoke test C: compute a "football" concept vector and sanity-check it.

Method (chosen empirically; see notes/smoke_tests.md): "word_tokens" extraction —
mean residual activation at the concept word's own tokens across sentence
templates, minus the same for random words. The two alternatives from the paper
sketch (last-prompt-token, response-mean) were tested and FAILED sanity checks:
they encode chat-template structure, not the concept.

Sanity metric: cosine between the vector and probe-token activations centered
by a generic-corpus mean (uncentered cosines are dominated by a shared generic
direction). Expect: high on the word "football", elevated text-mean on football
text without the word, ~0 on unrelated text.
"""

from irc import env  # noqa: F401

import json
import time

import torch
import torch.nn.functional as F

from irc.concept_vectors import concept_vector_word_tokens
from irc.model import MODEL_ID, ResidualCapture, chat_ids, load_model, load_tokenizer
from irc.paths import ARTIFACTS
from irc.words import RANDOM_WORDS_V1, WORD_TEMPLATES_V1

CONCEPT = "football"
CACHE = ARTIFACTS / "concept_vectors"
LAYERS_SHOWN = [10, 15, 22, 31, 40, 47, 55]

CENTER_TEXTS = [
    "The committee approved the budget after a lengthy discussion.",
    "She painted the old fence a bright shade of blue last weekend.",
    "Clouds gathered over the valley as the hikers reached the summit.",
    "The train to the coast leaves from platform four every hour.",
]
PROBES = [
    ("lexical (has 'football')", "He talked about football all evening with his friends."),
    ("contextual (no 'football')", "The goalkeeper made a brilliant save during the penalty shootout."),
    ("unrelated 1", "The recipe calls for two cups of flour and a pinch of salt."),
    ("unrelated 2", "Interest rates were left unchanged by the central bank on Thursday."),
]


@torch.no_grad()
def token_acts(model, tokenizer, text: str, layers: list[int]):
    ids = chat_ids(tokenizer, text)
    with ResidualCapture(model, layers) as cap:
        model(ids)
    toks = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    return {l: cap.acts[l][0, 1:] for l in layers}, toks[1:]  # BOS excluded


def main() -> None:
    tokenizer = load_tokenizer()
    model = load_model()

    t0 = time.time()
    vec = concept_vector_word_tokens(
        model, tokenizer, CONCEPT, RANDOM_WORDS_V1, WORD_TEMPLATES_V1
    )
    print(f"word_tokens concept vector: {tuple(vec.shape)} in {time.time() - t0:.0f}s")

    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(vec, CACHE / f"{CONCEPT}_word_tokens_v1.pt")
    (CACHE / f"{CONCEPT}_word_tokens_v1.json").write_text(json.dumps({
        "concept": CONCEPT,
        "model_id": MODEL_ID,
        "method": "word_tokens mean-difference",
        "random_words_version": "RANDOM_WORDS_V1",
        "templates_version": "WORD_TEMPLATES_V1",
        "capture": "resid_post all layers, fp32 on cpu, BOS excluded",
    }, indent=2))

    mu_parts = {l: [] for l in LAYERS_SHOWN}
    for t in CENTER_TEXTS:
        acts, _ = token_acts(model, tokenizer, t, LAYERS_SHOWN)
        for l in LAYERS_SHOWN:
            mu_parts[l].append(acts[l].mean(0))
    mu = {l: torch.stack(x).mean(0) for l, x in mu_parts.items()}

    for name, text in PROBES:
        acts, toks = token_acts(model, tokenizer, text, LAYERS_SHOWN)
        cells = []
        for l in LAYERS_SHOWN:
            c = F.cosine_similarity(acts[l] - mu[l], vec[l].unsqueeze(0), dim=-1)
            top = torch.topk(c, 1)
            tname = toks[top.indices[0]].replace("▁", "_").replace("\n", "NL")
            cells.append(f"L{l}: max={top.values[0]:+.2f}({tname}) mean={c.mean():+.2f}")
        print(f"\n{name}\n  " + " | ".join(cells))


if __name__ == "__main__":
    main()
