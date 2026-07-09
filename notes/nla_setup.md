# NLA (Natural Language Autoencoder) setup

Goal: decode stored response-token activations into natural-language
explanations with Anthropic's NLA actor ("AV", activation verbalizer) for
gemma-3-27b, to compare what the model represents in `think` vs `dont_think`
prompts.

Everything runtime-relevant is committed in-repo; the clones in `scratch/`
(`nla-inference`, `natural_language_autoencoders`) are reference only:
- client: vendored at `irc/vendor/nla_inference.py` (upstream commit +
  local fix documented in its header), used by `scripts/nla_explain.py`
- server env: `nla_server/` — a locked uv sub-project (Python 3.12 +
  sglang 0.5.8.post1) with the SGLang patches vendored in
  `nla_server/patches/`; `setup.sh` builds `.venv`, `launch.sh` serves

## Pieces

- **AV checkpoint**: `kitft/nla-gemma3-27b-L41-av` (108 GB, fp32 shards).
  Downloaded to `/workspace/hf-cache` — NOT the `.env` cache
  (`HF_HOME=/root/hf-cache`): the root disk holds the base gemma-3-27b-it
  (52 GB) and does not have another 108 GB. Snapshot path (pinned in
  `scripts/nla_explain.py`):
  `/workspace/hf-cache/hub/models--kitft--nla-gemma3-27b-L41-av/snapshots/4e721238131ffb8348cff260fe81b8b34a270a0d`
- **Sidecar** (`nla_meta.yaml`): d_model=5376, injection_scale=60000,
  injection char `㈜` (id 246566), extraction_layer_index=41.
- **Server env**: `nla_server/` — locked uv sub-project (Python 3.12,
  sglang 0.5.8.post1 + torch 2.9.1+cu128; sglang does not run on the root
  project's 3.14). `setup.sh` builds `.venv` (~10 GB, gitignored) and applies
  the 7 vendored NLA patches to the installed package via a shim; all anchors
  matched on 0.5.8. The gemma3_mm patch is load-bearing (without it the
  multimodal wrapper silently drops `input_embeds`); the rest are
  perf/robustness. flashinfer JIT-compiles sm_120 (Blackwell) kernels at
  first launch — needs ninja (in the venv) and nvcc (`/usr/local/cuda`) on
  PATH; `launch.sh` sets both.

## Running

```bash
bash nla_server/setup.sh    # once per machine (idempotent)
bash nla_server/launch.sh   # serve on :30000 (~54 GB VRAM, ready in minutes)

# Decode stored activations (root project venv, CPU-only client)
uv run python scripts/nla_explain.py --run-id run1-core \
    --words Dust --conditions think dont_think no_mention --limit 3
```

`--disable-radix-cache` is mandatory (radix cache keys on token ids, which
`input_embeds` requests don't have). `--dtype bfloat16` because the shards
are fp32 (108 GB > 96 GB VRAM); the actor was trained in bf16 anyway.

## Troubleshooting: stale flashinfer JIT cache across venvs

Symptom: the server dies at startup during "Capture cuda graph" with a ninja
build failure whose root cause is
`fatal error: flashinfer/attention/decode.cuh: No such file or directory`
(the headers exist; the compiler just isn't pointed at them).

Cause: flashinfer JIT-compiles Blackwell (`sm_120`, dir `120a`) attention
kernels on first launch and caches them under `/root/.cache/flashinfer`. That
cache is keyed **only by flashinfer version + GPU arch** (e.g. `0.6.1/120a`),
**not by venv path**. Each cached `build.ninja` bakes absolute `-isystem`
include paths into the venv that produced it. So if `nla_server/.venv` is
rebuilt (re-running `setup.sh`), or a *different* venv previously populated the
cache, the stale `build.ninja` still points `-isystem` at the old venv's
`site-packages/flashinfer/data/include`; when a kernel needs a (re)compile,
ninja can't find the headers and exits 1. Hit once (2026-07-09) after the cache
was first built from `/root/nla-venv`, then the project venv was rebuilt at
`nla_server/.venv`.

Fix: delete the cache — it is pure regenerable compile output, ~16 MB partial
(a full build is larger; measure with `du -sh` after a clean launch):

```bash
rm -rf /root/.cache/flashinfer      # recompiles on next launch (a few minutes)
```

`launch.sh` now does this automatically: a preflight greps the cached
`build.ninja` files and, if a baked flashinfer include path no longer exists,
clears the cache before starting (prints a warning). The cache is kept on the
root disk (`/root`, ~18 GB free) on purpose — **do not** relocate it to a
venv-local path via `FLASHINFER_CACHE_DIR`, because `/workspace` sits at ~99%
(it holds the 108 GB AV checkpoint).

## Layer indexing (important)

NLA `layer_index=41` = **output of decoder block 41** (resid_post), per
`nla/datagen/extractors.py` in the training repo. Our `ResidualCapture`
(`irc/model.py`) hooks decoder-layer outputs with the same 0-based indexing,
so **`acts[41]` is the right slice** — no ±1 offset. Stored acts are response
tokens only (no BOS position), but can include trailing whitespace token(s)
(the exact-match check strips the decoded text, so `sentence + "\n"` passes
with an extra token); `nla_explain.py` truncates to the sentence's token
count, same as `export_viz_data.py`. The client rescales every vector to
L2=60000 before injection, so bf16 storage / norm differences don't matter
beyond direction.

## Local fix to nla-inference (carried in the vendored copy)

Upstream `nla_inference.py` needed `return_dict=False` added to both
`apply_chat_template` calls: transformers ≥4.57 returns a `BatchEncoding`
by default, whose iteration yields dict keys, so the injection-position scan
found 0 sites (loud assert, caught at startup). The fix lives in
`irc/vendor/nla_inference.py` (see its provenance header). With it, all
sidecar asserts pass (injection site at the expected `>㈜<` neighbors
236813/954).

## Temperature: 0 (greedy) by default

`scripts/nla_explain.py` decodes at temperature 0. Rationale: every other
model call in the pipeline is greedy (`do_sample=False`), and upstream's
worked example for this exact checkpoint
(`scratch/nla-inference/examples/gemma27b_layer41_step6000.txt`) uses temp=0,
annotated "greedy — reproducible". The 0.7 the script previously used was
just the upstream CLI's argparse default, not a trained/eval'd value — at
0.7, repeat decodes of the same vector differ enough that think/dont_think
wording differences can't be separated from sampling noise. For robustness
analyses ("concept appears in k of n samples per vector"), sample repeats at
temperature 1.0 instead — that's the RL rollout distribution the actor was
trained under; 0.7 is neither that nor reproducible.

## Viewer integration

`scripts/export_viz_data.py` picks up `results/nla_explanations_token_L41.jsonl`
(per-token rows only) and attaches, per condition entry, the explanation texts
plus a binary `mention` flag: 1 if the target word appears verbatim
(whole-word, case-insensitive) in that token's explanation. no_mention
explanations are word-independent and only attached to words that have their
own NLA rows on that sentence. The viewer (`docs/index.html`) gets a fourth
measurement, "NLA explanation — word mentioned": binary 0/1 chart at fixed
layer 41 (slider disabled), with collapsed-by-default per-condition sections
below the chart showing each token's full explanation (mention rows
highlighted).

First data point (Secrecy s36, greedy): **0 verbatim mentions in all three
conditions** — but the think-condition explanations are saturated with
secrecy *semantics* ("clandestine", "concealing", "hush", "forbidden",
"secretive") that are absent from dont_think/no_mention, which decode as
generic grammar-lesson descriptions. The exact-word binary is a very
conservative detector; a synonym/semantic-set match would likely separate
the conditions here.

## Verification (2026-07-09, run1-core, layer 41, mean over response tokens)

Decoded `Dust` s27/s38/s39 across think / dont_think / no_mention at temp 0.7
(scratchpad `nla_verify.jsonl`). All outputs fluent English (no CJK soup →
injection works) and correctly describe the actual context: a repeated
sentence-completion task, quoting the experiment sentences almost verbatim.
Concept signal: `think__Dust__s27` mentions "the dust-word puzzle" and "Her
dust drifted"; neither dont_think nor no_mention decodes mention dust
(they read as "grammar instructional pattern..."). L41 response-token norms
~40k vs injection_scale 60k — in-distribution.
