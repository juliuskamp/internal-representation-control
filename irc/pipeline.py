"""Pipeline stages for the intentional-control replication.

Stages (each cached/resumable):
  vectors   — concept-vector banks (paper + word_tokens variants) for the 50
              concept words and 100 control words.
  generate  — for each (word, sentence) pair x condition: greedy generation,
              exact-output check, all-layer residual capture on the response
              sentence tokens. Non-exact outputs are flagged and excluded.
  latents   — per concept word: data-driven SAE latent selection at the
              Gemma Scope 2 layers, with Neuronpedia label cross-check.
  measure   — cosine of response-token activations with concept vectors
              (target word + 100 control words as null), and selected-latent
              SAE activations. No model needed; reads stored activations.
"""

from irc import env  # noqa: F401

import json
import random
import time
from pathlib import Path

import torch

from irc.conditions import WORD_FREE_CONDITIONS, build_prompt
from irc.concept_vectors import _word_token_span, build_vector_bank
from irc.constants import SAE_ID_TEMPLATE, SAE_RELEASE
from irc.model import ResidualCapture, chat_ids, get_decoder_layers
from irc.neuronpedia import fetch_label
from irc.paths import ARTIFACTS
from irc.words import WORD_TEMPLATES_V1
from irc.words_paper import (
    BASELINE_WORDS_PAPER,
    CONCEPT_WORDS_PAPER,
    CONTROL_WORDS_PAPER,
    SENTENCES_PAPER,
)


# ---------------------------------------------------------------- vectors ----

def ensure_vector_bank(model, tokenizer, variant: str) -> Path:
    """Compute (or reuse cached) concept vectors for concept + control words."""
    path = ARTIFACTS / "concept_vectors" / f"bank_{variant}_v1.pt"
    if path.exists():
        print(f"[vectors] cached: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[vectors] building variant {variant!r} "
          f"({len(CONCEPT_WORDS_PAPER)} concept + {len(CONTROL_WORDS_PAPER)} control words)")
    bank = build_vector_bank(
        model,
        tokenizer,
        variant,
        words=CONCEPT_WORDS_PAPER + CONTROL_WORDS_PAPER,
        baseline_words=BASELINE_WORDS_PAPER,
        templates=WORD_TEMPLATES_V1 if variant == "word_tokens" else None,
    )
    bank["meta"] = {
        "variant": variant,
        "concept_words": CONCEPT_WORDS_PAPER,
        "control_words": CONTROL_WORDS_PAPER,
        "n_baseline_words": len(BASELINE_WORDS_PAPER),
        "templates": WORD_TEMPLATES_V1 if variant == "word_tokens" else "Tell me about {word}.",
    }
    torch.save(bank, path)
    print(f"[vectors] saved {path}")
    return path


# ------------------------------------------------------------------ pairs ----

def pair_table(
    words: list[str], sentences_per_word: int, seed: int
) -> list[tuple[str, int]]:
    """Deterministic (word, sentence_idx) pairs; increasing sentences_per_word
    extends the set without changing earlier pairs."""
    pairs = []
    for w in words:
        order = random.Random(f"{seed}-{w}").sample(
            range(len(SENTENCES_PAPER)), len(SENTENCES_PAPER)
        )
        pairs.extend((w, si) for si in order[:sentences_per_word])
    return pairs


# --------------------------------------------------------------- generate ----

def record_key(condition: str, word: str | None, si: int) -> str:
    return f"{condition}__{word or 'NONE'}__s{si:02d}"


@torch.no_grad()
def _generate_and_capture(model, tokenizer, prompt: str, sentence: str) -> dict:
    """Greedy generation + exact check + all-layer capture on response tokens."""
    ids = chat_ids(tokenizer, prompt)
    n_prompt = ids.shape[1]
    sent_len = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
    out = model.generate(ids, max_new_tokens=sent_len + 16, do_sample=False)
    gen = out[0, n_prompt:]

    end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ends = (gen == end_id).nonzero()
    n_resp = int(ends[0]) if len(ends) else len(gen)
    completion = tokenizer.decode(gen[:n_resp], skip_special_tokens=True).strip()

    exact = completion == sentence
    paper_match = sentence.lower() in completion.lower()
    result = {
        "prompt": prompt,
        "completion": completion,
        "exact_match": exact,
        "paper_match": paper_match,
        "n_resp_tokens": n_resp,
        "acts": None,
    }
    if exact:
        layers = list(range(len(get_decoder_layers(model))))
        with ResidualCapture(model, layers) as cap:
            model(out[:, : n_prompt + n_resp])
        acts = torch.stack([cap.acts[i][0, n_prompt : n_prompt + n_resp] for i in layers])
        result["acts"] = acts.to(torch.bfloat16)  # (n_layers, T, d_model)
    return result


def run_generations(
    model, tokenizer, run_dir: Path, pairs: list[tuple[str, int]], conditions: list[str]
) -> None:
    acts_dir = run_dir / "acts"
    acts_dir.mkdir(parents=True, exist_ok=True)
    gen_path = run_dir / "generations.jsonl"
    done = {}
    if gen_path.exists():
        with gen_path.open() as f:
            done = {r["key"]: r for r in map(json.loads, f)}

    # word-free conditions run once per unique sentence
    jobs: list[tuple[str, str | None, int]] = []
    seen = set()
    for condition in conditions:
        for word, si in pairs:
            w = None if condition in WORD_FREE_CONDITIONS else word
            key = record_key(condition, w, si)
            if key in seen or key in done:
                continue
            seen.add(key)
            jobs.append((condition, w, si))

    print(f"[generate] {len(jobs)} generations to run ({len(done)} already done)")
    n_flagged = 0
    with gen_path.open("a") as f:
        for i, (condition, word, si) in enumerate(jobs):
            sentence = SENTENCES_PAPER[si]
            prompt = build_prompt(condition, sentence, word)
            res = _generate_and_capture(model, tokenizer, prompt, sentence)
            key = record_key(condition, word, si)
            acts_file = None
            if res["acts"] is not None:
                acts_file = f"acts/{key}.pt"
                torch.save(res["acts"], run_dir / acts_file)
            else:
                n_flagged += 1
                print(f"  FLAGGED (not exact) {key}: {res['completion']!r}")
            record = {
                "key": key,
                "condition": condition,
                "word": word,
                "sentence_idx": si,
                "sentence": sentence,
                "prompt": res["prompt"],
                "completion": res["completion"],
                "exact_match": res["exact_match"],
                "paper_match": res["paper_match"],
                "n_resp_tokens": res["n_resp_tokens"],
                "acts_file": acts_file,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  [generate] {i + 1}/{len(jobs)}")
    print(f"[generate] done; {n_flagged} flagged as non-exact this session")


# ---------------------------------------------------------------- latents ----

def _neuronpedia_label(layer: int, index: int, cache: dict, cache_path: Path) -> str:
    key = f"{layer}/{index}"
    if key in cache:
        return cache[key]
    try:
        label = fetch_label(layer, index)
    except Exception as e:
        return f"(lookup failed: {e})"  # not cached — may be transient
    cache[key] = label
    cache_path.write_text(json.dumps(cache, indent=1))
    return label


def load_saes(sae_layers: list[int], device: str = "cuda", total_timeout: float = 300.0) -> dict:
    """Load the pinned SAEs, retrying on transient network errors.

    sae_lens fetches the safetensors header over HTTP on every load (hardcoded
    10s timeout) regardless of local cache, so a flaky CDN can kill an otherwise
    model-free measure. Retry each layer with exponential backoff, bounded by a
    total wall-clock budget shared across all layers."""
    from sae_lens import SAE

    env.require_hf_token()
    saes = {}
    deadline = time.monotonic() + total_timeout
    for layer in sae_layers:
        delay = 2.0
        while True:
            try:
                r = SAE.from_pretrained(
                    SAE_RELEASE, SAE_ID_TEMPLATE.format(layer=layer), device=device
                )
                saes[layer] = r[0] if isinstance(r, tuple) else r
                break
            except Exception as e:  # noqa: BLE001 — retry any transient failure
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                wait = min(delay, remaining)
                print(f"[measure] SAE layer {layer} load failed ({e}); "
                      f"retrying in {wait:.0f}s ({remaining:.0f}s budget left)")
                time.sleep(wait)
                delay = min(delay * 2, 30.0)
    return saes


@torch.no_grad()
def _sae_feats(
    model, tokenizer, saes: dict, sae_layers: list[int], text: str, word: str | None = None
) -> dict[int, torch.Tensor]:
    """SAE features per layer for a chat-templated prompt; if `word` is given,
    only the word's own token positions (avoids selecting latents for the
    template phrasing), else all positions with BOS excluded."""
    if word is None:
        ids = chat_ids(tokenizer, text)
        span = slice(1, None)  # BOS excluded
    else:
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False,
        )
        ids = tokenizer(full, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)
        span = _word_token_span(tokenizer, ids[0], full, word)
    with ResidualCapture(model, sae_layers) as cap:
        model(ids)
    return {
        l: saes[l].encode(cap.acts[l][0, span].to("cuda", saes[l].dtype))
        for l in sae_layers
    }  # (T, d_sae) each


@torch.no_grad()
def _word_template_mean(
    model, tokenizer, saes: dict, sae_layers: list[int], word: str
) -> dict[int, torch.Tensor]:
    """Per layer: mean SAE activation over the word's own token span, averaged
    over the 4 templates — the `concept_mean` statistic (same in v1 and v2)."""
    w = word.lower()
    sums = {l: None for l in sae_layers}
    for t in WORD_TEMPLATES_V1:
        f = _sae_feats(model, tokenizer, saes, sae_layers, t.format(word=w), word=w)
        for l in sae_layers:
            m = f[l].mean(dim=0)
            sums[l] = m if sums[l] is None else sums[l] + m
    return {l: (sums[l] / len(WORD_TEMPLATES_V1)).float().cpu() for l in sae_layers}


@torch.no_grad()
def select_latents(
    model, tokenizer, words: list[str], sae_layers: list[int], topk: int,
    neuronpedia: bool, version: str = "v1",
) -> Path:
    """Per word and SAE layer: select concept-selective latents.

    v1 — rank by raw concept_mean (word-token-span activation across the 4
    templates); exclude latents with non-negligible max activation on the 50
    experiment sentences.
    v2 — rank by concept_mean minus the mean of the same statistic over the 99
    paper BASELINE words (contrastive, kills generic/template latents); exclude
    on the 100 CONTROL words' template prompts (all tokens, BOS excluded)
    instead of the experiment sentences, which selection never sees.
    """
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown latents version {version!r}")
    out_dir = ARTIFACTS / f"latents_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saes = load_saes(sae_layers)
    np_cache_path = ARTIFACTS / "neuronpedia_cache.json"
    np_cache = json.loads(np_cache_path.read_text()) if np_cache_path.exists() else {}
    if version == "v2":
        return _select_latents_v2(
            model, tokenizer, saes, out_dir, words, sae_layers, topk,
            neuronpedia, np_cache, np_cache_path,
        )

    def feats(text: str, word: str | None = None) -> dict[int, torch.Tensor]:
        return _sae_feats(model, tokenizer, saes, sae_layers, text, word)

    base_path = out_dir / "_baseline_max.pt"
    if base_path.exists():
        baseline_max = torch.load(base_path)
    else:
        print("[latents] computing baseline max over 50 sentences")
        baseline_max = {l: None for l in sae_layers}
        for s in SENTENCES_PAPER:
            f = feats(s)
            for l in sae_layers:
                m = f[l].max(dim=0).values
                baseline_max[l] = m if baseline_max[l] is None else torch.maximum(baseline_max[l], m)
        torch.save(baseline_max, base_path)

    for wi, word in enumerate(words):
        path = out_dir / f"{word}.json"
        if path.exists():
            continue
        concept_means = _word_template_mean(model, tokenizer, saes, sae_layers, word)
        entry = {"word": word, "sae_release": SAE_RELEASE, "layers": {}}
        for l in sae_layers:
            concept_mean = concept_means[l]
            bmax = baseline_max[l].float().cpu()
            eligible = bmax < 0.1 * concept_mean.clamp(min=1e-6)
            score = torch.where(eligible, concept_mean, torch.zeros_like(concept_mean))
            top = torch.topk(score, topk)
            latents = []
            for idx, val in zip(top.indices.tolist(), top.values.tolist()):
                if val <= 0:
                    continue
                label = (_neuronpedia_label(l, idx, np_cache, np_cache_path)
                         if neuronpedia else "")
                latents.append({
                    "latent": idx,
                    "concept_mean": round(val, 3),
                    "baseline_max": round(float(bmax[idx]), 3),
                    "label": label,
                })
            entry["layers"][str(l)] = latents
        path.write_text(json.dumps(entry, indent=1))
        if (wi + 1) % 10 == 0:
            print(f"  [latents] {wi + 1}/{len(words)}")
    return out_dir


@torch.no_grad()
def _select_latents_v2(
    model, tokenizer, saes: dict, out_dir: Path, words: list[str],
    sae_layers: list[int], topk: int, neuronpedia: bool,
    np_cache: dict, np_cache_path: Path,
) -> Path:
    """v2 selection body (see select_latents). The 50 experiment sentences are
    never used here — the no_mention floor in measurement stays an empirical
    result rather than holding by construction."""
    # Per-word template statistic, cached incrementally (concept + baseline words).
    means_path = out_dir / "_word_template_means.pt"
    word_means: dict = torch.load(means_path) if means_path.exists() else {}
    todo = [w for w in dict.fromkeys(list(words) + BASELINE_WORDS_PAPER)
            if w not in word_means]
    if todo:
        print(f"[latents v2] template means for {len(todo)} words "
              f"({len(word_means)} cached)")
        for i, w in enumerate(todo):
            word_means[w] = _word_template_mean(model, tokenizer, saes, sae_layers, w)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                torch.save(word_means, means_path)
                print(f"  [latents v2] means {i + 1}/{len(todo)}")

    # Exclusion stat: max over ALL token positions (BOS excluded) of the 100
    # control words' template prompts — all-token on purpose, so latents firing
    # on the template phrasing itself get excluded too.
    ctrl_path = out_dir / "_control_max.pt"
    if ctrl_path.exists():
        control_max = torch.load(ctrl_path)
    else:
        print(f"[latents v2] control max over {len(CONTROL_WORDS_PAPER)} words "
              f"x {len(WORD_TEMPLATES_V1)} templates (all tokens)")
        control_max = {l: None for l in sae_layers}
        for i, w in enumerate(CONTROL_WORDS_PAPER):
            for t in WORD_TEMPLATES_V1:
                f = _sae_feats(model, tokenizer, saes, sae_layers,
                               t.format(word=w.lower()))
                for l in sae_layers:
                    m = f[l].max(dim=0).values
                    control_max[l] = (m if control_max[l] is None
                                      else torch.maximum(control_max[l], m))
            if (i + 1) % 25 == 0:
                print(f"  [latents v2] control {i + 1}/{len(CONTROL_WORDS_PAPER)}")
        control_max = {l: v.float().cpu() for l, v in control_max.items()}
        torch.save(control_max, ctrl_path)

    baseline_mean = {
        l: torch.stack([word_means[b][l] for b in BASELINE_WORDS_PAPER]).mean(dim=0)
        for l in sae_layers
    }

    for wi, word in enumerate(words):
        path = out_dir / f"{word}.json"
        if path.exists():
            continue
        entry = {
            "word": word,
            "sae_release": SAE_RELEASE,
            "selection": "v2",  # contrastive score, control-word exclusion
            "layers": {},
        }
        for l in sae_layers:
            cm = word_means[word][l]
            bwm = baseline_mean[l]
            cmax = control_max[l]
            # Same threshold form as v1, referenced to raw concept_mean.
            eligible = cmax < 0.1 * cm.clamp(min=1e-6)
            contrast = cm - bwm
            score = torch.where(eligible, contrast, torch.zeros_like(contrast))
            top = torch.topk(score, topk)
            latents = []
            for idx, val in zip(top.indices.tolist(), top.values.tolist()):
                if val <= 0:
                    continue
                label = (_neuronpedia_label(l, idx, np_cache, np_cache_path)
                         if neuronpedia else "")
                latents.append({
                    "latent": idx,
                    "concept_mean": round(float(cm[idx]), 3),
                    "contrast_score": round(val, 3),
                    "baseline_word_mean": round(float(bwm[idx]), 3),
                    "baseline_max": round(float(cmax[idx]), 3),
                    "label": label,
                })
            entry["layers"][str(l)] = latents
        path.write_text(json.dumps(entry, indent=1))
        if (wi + 1) % 10 == 0:
            print(f"  [latents v2] {wi + 1}/{len(words)}")
    return out_dir


# ---------------------------------------------------------------- measure ----

def load_records(run_dir: Path) -> list[dict]:
    with (run_dir / "generations.jsonl").open() as f:
        return [json.loads(line) for line in f]


def load_vector_bank(variant: str, device: str = "cuda") -> dict:
    """Normalized concept-vector bank for measurement:
    {"w_idx": word -> row, "ctrl_idx": control-word rows (CONTROL_WORDS_PAPER
    order), "Vn": (W, L, D) unit vectors on `device`}."""
    bank = torch.load(ARTIFACTS / "concept_vectors" / f"bank_{variant}_v1.pt")
    word_list = list(bank["vectors"].keys())
    w_idx = {w: i for i, w in enumerate(word_list)}
    V = torch.stack([bank["vectors"][w] for w in word_list]).to(device)
    return {
        "w_idx": w_idx,
        "ctrl_idx": torch.tensor([w_idx[w] for w in CONTROL_WORDS_PAPER]),
        "Vn": V / V.norm(dim=-1, keepdim=True),
    }


def concept_cosines(A: torch.Tensor, Vn: torch.Tensor) -> torch.Tensor:
    """Cosine of activations (L, T, D) with unit concept vectors (W, L, D)
    -> (L, W, T)."""
    An = A / A.norm(dim=-1, keepdim=True)
    return torch.einsum("ltd,wld->lwt", An, Vn)


@torch.no_grad()
def measure(
    run_dir: Path,
    pairs: list[tuple[str, int]],
    conditions: list[str],
    variants: list[str],
    sae_layers: list[int],
    device: str = "cuda",
    latents_version: str = "v1",
) -> None:
    import pandas as pd

    records = {r["key"]: r for r in load_records(run_dir)}
    results_dir = run_dir / "results"
    (results_dir / "token_cosines").mkdir(parents=True, exist_ok=True)
    (results_dir / "null_means").mkdir(parents=True, exist_ok=True)

    def acts_for(condition: str, word: str, si: int) -> torch.Tensor | None:
        w = None if condition in WORD_FREE_CONDITIONS else word
        rec = records.get(record_key(condition, w, si))
        if rec is None or not rec["acts_file"]:
            return None
        return torch.load(run_dir / rec["acts_file"]).float().to(device)

    # ---- concept vectors (independent of latents_version; pass variants=[]
    # to skip, e.g. when re-measuring only under a new latent set)
    rows = []
    for variant in variants:
        bank = load_vector_bank(variant, device)
        w_idx, ctrl_idx, Vn = bank["w_idx"], bank["ctrl_idx"], bank["Vn"]
        for word, si in pairs:
            for condition in conditions:
                A = acts_for(condition, word, si)
                if A is None:
                    continue
                cos = concept_cosines(A, Vn)  # (L, W, T)
                tok_mean, tok_max = cos.mean(-1), cos.max(-1).values  # (L, W)
                tgt = w_idx[word]
                null_mean = tok_mean[:, ctrl_idx]
                key = record_key(condition, word, si)
                torch.save(cos[:, tgt].cpu(),
                           results_dir / "token_cosines" / f"{variant}__{key}.pt")
                # (n_layers, n_control_words) token-mean cosines, so plots can
                # form the paper-style band: spread across control words of the
                # pair-averaged cosine.
                torch.save(null_mean.half().cpu(),
                           results_dir / "null_means" / f"{variant}__{key}.pt")
                for layer in range(cos.shape[0]):
                    rows.append({
                        "variant": variant, "word": word, "sentence_idx": si,
                        "condition": condition, "layer": layer,
                        "cos_mean": float(tok_mean[layer, tgt]),
                        "cos_max": float(tok_max[layer, tgt]),
                        "null_mean": float(null_mean[layer].mean()),
                        "null_std": float(null_mean[layer].std()),
                        "null_q95": float(null_mean[layer].quantile(0.95)),
                        "null_q05": float(null_mean[layer].quantile(0.05)),
                    })
    if variants:
        df = pd.DataFrame(rows)
        df.to_parquet(results_dir / "concept_cosines.parquet")
        print(f"[measure] concept cosines: {len(df)} rows -> results/concept_cosines.parquet")

    # ---- SAE latents
    latents_dir = ARTIFACTS / f"latents_{latents_version}"
    saes = load_saes(sae_layers, device=device)
    sae_rows = []
    for word, si in pairs:
        lat_path = latents_dir / f"{word}.json"
        if not lat_path.exists():
            continue
        sel = json.loads(lat_path.read_text())["layers"]
        for condition in conditions:
            A = acts_for(condition, word, si)
            if A is None:
                continue
            for layer in sae_layers:
                latents = [e["latent"] for e in sel.get(str(layer), [])]
                if not latents:
                    continue
                feats = saes[layer].encode(A[layer].to(saes[layer].dtype))  # (T, d_sae)
                f_sel = feats[:, latents].float()
                sae_rows.append({
                    "word": word, "sentence_idx": si, "condition": condition,
                    "layer": layer, "latents": json.dumps(latents),
                    "act_mean": float(f_sel.mean()),
                    "act_max": float(f_sel.max()),
                    "act_sum_mean": float(f_sel.sum(dim=1).mean()),
                    "frac_tokens_active": float((f_sel.max(dim=1).values > 0).float().mean()),
                })
    sdf = pd.DataFrame(sae_rows)
    # v1 keeps its historical filename; other versions get their own file so
    # re-measuring under a new latent set never clobbers earlier results.
    sae_name = ("sae_latents.parquet" if latents_version == "v1"
                else f"sae_latents_{latents_version}.parquet")
    sdf.to_parquet(results_dir / sae_name)
    print(f"[measure] SAE latents ({latents_version}): {len(sdf)} rows -> results/{sae_name}")

    flagged = [r for r in records.values() if not r["exact_match"]]
    summary = {
        "n_records": len(records),
        "n_flagged_not_exact": len(flagged),
        "flagged": [{"key": r["key"], "completion": r["completion"]} for r in flagged],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[measure] {len(flagged)} flagged non-exact generations (see results/summary.json)")
