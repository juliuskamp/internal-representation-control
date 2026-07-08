"""Concept vectors via the mean-difference method (paper-faithful).

For a concept word w: vector(layer) = mean_act(prompts about w) - mean_act(prompts
about random words). Two position variants are computed in the same pass:
  - "last_prompt": activation at the final prompt token (end of chat template).
  - "response_mean": mean activation over greedily generated response tokens.
"""

from irc import env  # noqa: F401

import torch

from irc.model import ResidualCapture, chat_ids, get_decoder_layers

POSITION_VARIANTS = ("last_prompt", "response_mean")


@torch.no_grad()
def prompt_activations(
    model,
    tokenizer,
    user_message: str,
    layers: list[int],
    max_new_tokens: int = 32,
) -> dict[str, torch.Tensor]:
    """Run one prompt, return {variant: (n_layers, d_model)} fp32 cpu tensors."""
    ids = chat_ids(tokenizer, user_message)
    n_prompt = ids.shape[1]
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    with ResidualCapture(model, layers) as cap:
        model(out)
    last_prompt = torch.stack([cap.acts[i][0, n_prompt - 1] for i in layers])
    response_mean = torch.stack([cap.acts[i][0, n_prompt:].mean(dim=0) for i in layers])
    return {"last_prompt": last_prompt, "response_mean": response_mean}


# Paper-faithful extraction (appendix 12.1.1): activations at the final prompt
# token of "Tell me about {word}." (word lowercase) — for Gemma's chat template
# this is the last token of the generation prompt, the analogue of the paper's
# final ":" of "Assistant:".
PAPER_TEMPLATE = "Tell me about {word}."


@torch.no_grad()
def last_token_activations(model, tokenizer, text: str, layers: list[int]) -> torch.Tensor:
    """(n_layers, d_model) fp32 cpu activation at the final prompt token."""
    ids = chat_ids(tokenizer, text)
    with ResidualCapture(model, layers) as cap:
        model(ids)
    return torch.stack([cap.acts[i][0, -1] for i in layers])


def build_vector_bank(
    model,
    tokenizer,
    variant: str,
    words: list[str],
    baseline_words: list[str],
    templates: list[str] | None = None,
    log_every: int = 25,
) -> dict:
    """Compute concept vectors for `words` (concept + control) in one variant.

    variant "paper": last-prompt-token of "Tell me about {word}.".
    variant "word_tokens": mean activation at the word's own tokens across
    `templates`.
    Returns {"vectors": {word: (n_layers, d_model)}, "baseline_mean": tensor,
    "raw": {word: tensor}} — all fp32 cpu; words keyed as given (capitalized).
    """
    layers = list(range(len(get_decoder_layers(model))))

    def acts_for(word: str) -> torch.Tensor:
        w = word.lower()
        if variant == "paper":
            return last_token_activations(
                model, tokenizer, PAPER_TEMPLATE.format(word=w), layers
            )
        if variant == "word_tokens":
            assert templates
            return torch.stack(
                [word_token_activations(model, tokenizer, t, w, layers) for t in templates]
            ).mean(dim=0)
        raise ValueError(f"unknown variant {variant!r}")

    baseline_sum = None
    for i, w in enumerate(baseline_words):
        a = acts_for(w)
        baseline_sum = a if baseline_sum is None else baseline_sum + a
        if (i + 1) % log_every == 0:
            print(f"  [{variant}] baseline {i + 1}/{len(baseline_words)}")
    baseline_mean = baseline_sum / len(baseline_words)

    raw = {}
    for i, w in enumerate(words):
        raw[w] = acts_for(w)
        if (i + 1) % log_every == 0:
            print(f"  [{variant}] words {i + 1}/{len(words)}")
    vectors = {w: a - baseline_mean for w, a in raw.items()}
    return {"vectors": vectors, "baseline_mean": baseline_mean, "raw": raw}


def _word_token_span(tokenizer, ids: torch.Tensor, full_text: str, word: str) -> slice:
    """Locate the token positions covering `word` in the tokenized chat string."""
    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    if list(enc["input_ids"]) != ids.tolist():
        raise ValueError("re-tokenization mismatch — offsets unusable")
    start_char = full_text.rindex(word)
    end_char = start_char + len(word)
    positions = [
        i
        for i, (s, e) in enumerate(enc["offset_mapping"])
        if s < end_char and e > start_char
    ]
    if not positions:
        raise ValueError(f"word {word!r} not found in offsets")
    return slice(positions[0], positions[-1] + 1)


@torch.no_grad()
def word_token_activations(
    model, tokenizer, template: str, word: str, layers: list[int]
) -> torch.Tensor:
    """(n_layers, d_model): mean resid activation over the word's own tokens."""
    text = template.format(word=word)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
    )
    ids = tokenizer(full, return_tensors="pt", add_special_tokens=False)["input_ids"]
    span = _word_token_span(tokenizer, ids[0], full, word)
    with ResidualCapture(model, layers) as cap:
        model(ids.to(model.device))
    return torch.stack([cap.acts[i][0, span].mean(dim=0) for i in layers])


@torch.no_grad()
def concept_vector_word_tokens(
    model,
    tokenizer,
    concept: str,
    random_words: list[str],
    templates: list[str],
    layers: list[int] | None = None,
) -> torch.Tensor:
    """(n_layers, d_model) concept vector from word-token activations."""
    if layers is None:
        layers = list(range(len(get_decoder_layers(model))))

    def word_mean(word: str) -> torch.Tensor:
        return torch.stack(
            [word_token_activations(model, tokenizer, t, word, layers) for t in templates]
        ).mean(dim=0)

    concept_acts = word_mean(concept)
    baseline = torch.stack([word_mean(w) for w in random_words]).mean(dim=0)
    return concept_acts - baseline


@torch.no_grad()
def concept_vector(
    model,
    tokenizer,
    concept: str,
    random_words: list[str],
    templates: list[str],
    layers: list[int] | None = None,
    max_new_tokens: int = 32,
) -> dict[str, torch.Tensor]:
    """Return {variant: (n_layers, d_model)} concept vectors (all layers if None)."""
    if layers is None:
        layers = list(range(len(get_decoder_layers(model))))

    def word_mean(word: str) -> dict[str, torch.Tensor]:
        per_template = [
            prompt_activations(model, tokenizer, t.format(word=word), layers, max_new_tokens)
            for t in templates
        ]
        return {
            v: torch.stack([p[v] for p in per_template]).mean(dim=0)
            for v in POSITION_VARIANTS
        }

    concept_acts = word_mean(concept)
    baseline_sum = {v: torch.zeros_like(concept_acts[v]) for v in POSITION_VARIANTS}
    for w in random_words:
        acts = word_mean(w)
        for v in POSITION_VARIANTS:
            baseline_sum[v] += acts[v]
    return {
        v: concept_acts[v] - baseline_sum[v] / len(random_words)
        for v in POSITION_VARIANTS
    }
