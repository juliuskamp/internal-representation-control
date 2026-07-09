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
