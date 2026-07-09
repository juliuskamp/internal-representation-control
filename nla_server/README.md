# nla_server

Locked serving environment for the NLA activation verbalizer
(`kitft/nla-gemma3-27b-L41-av`) — see `notes/nla_setup.md` for the full
context and `scripts/nla_explain.py` for the client side.

This is a separate uv project (own `pyproject.toml` + `uv.lock`) because
sglang requires Python 3.12 while the root project runs 3.14. The venv itself
(`.venv/`, ~10 GB) is gitignored; only the definition is committed.

```bash
bash nla_server/setup.sh    # create .venv, apply patches (idempotent)
bash nla_server/launch.sh   # serve on localhost:30000
```

`patches/` is vendored from
https://github.com/kitft/natural_language_autoencoders
(commit 1b7f13d9d8a37075cd2e5d1604eca57820216ed5, Apache-2.0 — LICENSE
alongside). The `gemma3_mm` patch is load-bearing: without it SGLang's
multimodal wrapper silently drops `input_embeds` and the injection never
happens. The rest are perf/robustness fixes. All anchors verified against
sglang 0.5.8.post1.
