# nla_server

Locked serving environment for the NLA activation verbalizer
(`kitft/nla-gemma3-27b-L41-av`) — see `scripts/nla_explain.py` for the client
side.

This is a separate uv project (own `pyproject.toml` + `uv.lock`) because
sglang requires Python 3.12 while the root project runs 3.13. The venv itself
(`.venv/`, ~10 GB) is gitignored; only the definition is committed.

```bash
bash nla_server/setup.sh    # create .venv, apply patches (idempotent)
bash nla_server/launch.sh   # serve on localhost:30000
```

Needs room for the checkpoint (108 GB, fp32 shards) plus ~54 GB VRAM to
serve it; `launch.sh` loads it as bf16, which is what the actor was trained
in anyway. Point `$NLA_AV_CHECKPOINT` (or `launch.sh`'s first argument) at
the snapshot if it does not live under `$HF_HOME`.

flashinfer JIT-compiles its attention kernels on first launch, so `ninja`
(in the venv) and `nvcc` must be on PATH — `launch.sh` sets both. Its cache
is keyed by flashinfer version + GPU arch only, *not* by venv path, so a
rebuilt venv leaves stale absolute include paths behind and the server dies
during cuda-graph capture with `fatal error: flashinfer/attention/decode.cuh:
No such file or directory`. `launch.sh` detects that and clears the cache.

`patches/` is vendored from
https://github.com/kitft/natural_language_autoencoders
(commit 1b7f13d9d8a37075cd2e5d1604eca57820216ed5, Apache-2.0 — LICENSE
alongside). The `gemma3_mm` patch is load-bearing: without it SGLang's
multimodal wrapper silently drops `input_embeds` and the injection never
happens. The rest are perf/robustness fixes. All anchors verified against
sglang 0.5.8.post1.
