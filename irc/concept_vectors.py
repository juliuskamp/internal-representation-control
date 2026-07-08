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
