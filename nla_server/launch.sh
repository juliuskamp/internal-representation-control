#!/usr/bin/env bash
# Launch the SGLang server for the NLA AV model (foreground; ~54 GB VRAM,
# ready after a few minutes — watch for "The server is fired up").
#
# Usage: bash nla_server/launch.sh [model_snapshot_dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The AV checkpoint lives in /workspace/hf-cache, NOT the .env HF_HOME
# (/root/hf-cache) — the root disk cannot hold the 108 GB repo alongside the
# base model. See notes/nla_setup.md.
DEFAULT_SNAP=/workspace/hf-cache/hub/models--kitft--nla-gemma3-27b-L41-av/snapshots/4e721238131ffb8348cff260fe81b8b34a270a0d
SNAP="${1:-$DEFAULT_SNAP}"
[ -f "$SNAP/nla_meta.yaml" ] || { echo "error: $SNAP is not an NLA checkpoint (no nla_meta.yaml)" >&2; exit 1; }

# ninja (venv) + nvcc (system CUDA) must be on PATH for flashinfer's JIT.
export PATH="$HERE/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

# Preflight: bust a stale flashinfer JIT cache. The cache is keyed only by
# flashinfer version + GPU arch (e.g. 0.6.1/120a), NOT by venv path. If .venv
# was rebuilt (or a different venv previously populated the cache), the cached
# build.ninja files still carry -isystem include paths into the OLD venv, and
# ninja fails mid-launch during cuda-graph capture with:
#   fatal error: flashinfer/attention/decode.cuh: No such file or directory
# The cache is pure regenerable compile output, so if a baked include path no
# longer resolves, clear it and let this launch recompile (a few minutes).
# Kept on the root disk (/root, ~18 GB free) deliberately — /workspace is at
# ~99% (holds the 108 GB AV checkpoint). See notes/nla_setup.md.
FI_CACHE="${FLASHINFER_CACHE_DIR:-${HOME:-/root}/.cache/flashinfer}"
if [ -d "$FI_CACHE" ]; then
    stale_inc="$(grep -rhoE --include=build.ninja \
        'isystem +[^ ]*/site-packages/flashinfer/data/include' "$FI_CACHE" \
        2>/dev/null | awk '{print $2}' | head -1 || true)"
    if [ -n "$stale_inc" ] && [ ! -d "$stale_inc" ]; then
        echo "warning: flashinfer JIT cache is stale (built by a different/" >&2
        echo "  rebuilt venv). Baked include path no longer exists:" >&2
        echo "    $stale_inc" >&2
        echo "  clearing $FI_CACHE — kernels recompile on this launch." >&2
        rm -rf "$FI_CACHE"
    fi
fi

# --disable-radix-cache is REQUIRED (cache keys on token ids, which
# input_embeds requests don't have). --dtype bfloat16: shards are fp32.
exec "$HERE/.venv/bin/python" -m sglang.launch_server \
    --model-path "$SNAP" \
    --port 30000 \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    --dtype bfloat16 \
    --context-length 512 \
    --trust-remote-code
