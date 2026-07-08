"""Model loading and residual-stream activation capture for Gemma 3 27B-it."""

from irc import env  # noqa: F401  (must run before transformers import)

import torch
from torch import nn
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-27b-it"


def load_tokenizer(model_id: str = MODEL_ID):
    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str = MODEL_ID, device: str = "cuda"):
    """Load the model in bfloat16 (never fp32/quantized — we measure activations)."""
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
    except ValueError:
        # gemma-3-27b-it is a multimodal checkpoint; fall back to the
        # conditional-generation class and use its text stack.
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
    model.eval()
    return model


def get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return the list of decoder layers regardless of wrapper class."""
    for path in ("model.layers", "language_model.layers", "model.language_model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return list(obj)
    raise AttributeError(f"could not locate decoder layers on {type(model).__name__}")


class ResidualCapture:
    """Context manager capturing resid_post (decoder-layer outputs) at given layers.

    Activations are moved to CPU and upcast to fp32 inside the hook. After a
    forward pass, `self.acts[layer]` is (batch, seq, d_model).
    """

    def __init__(self, model: nn.Module, layers: list[int]):
        self.layers = layers
        self._decoder_layers = get_decoder_layers(model)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.acts: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.acts[layer_idx] = hidden.detach().float().cpu()

        return hook

    def __enter__(self):
        for i in self.layers:
            self._handles.append(
                self._decoder_layers[i].register_forward_hook(self._make_hook(i))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def chat_ids(tokenizer, user_message: str, device: str = "cuda") -> torch.Tensor:
    """Tokenize a single-turn user message with the chat template."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device)
