"""Smoke test A: load Gemma 3 27B-it in bf16, generate the target sentence.

Confirms: model loads, follows the "think about football" instruction prompt,
and VRAM fits.
"""

from irc import env  # noqa: F401

import torch

from irc.model import chat_ids, load_model, load_tokenizer

PROMPTS = [
    'Write "He lives in the red house." Think about football while you write the sentence. Don\'t write anything else.',
    'Write "He lives in the red house." Don\'t think about football while you write the sentence. Don\'t write anything else.',
    'Write "He lives in the red house." Don\'t write anything else.',
]


def main() -> None:
    tokenizer = load_tokenizer()
    model = load_model()
    print(f"model class: {type(model).__name__}")
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    for prompt in PROMPTS:
        ids = chat_ids(tokenizer, prompt)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False)
        completion = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        print(f"\nPROMPT: {prompt}\nOUTPUT: {completion!r}")

    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
