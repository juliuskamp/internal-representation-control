#!/usr/bin/env bash
# Build the NLA SGLang serving env at nla_server/.venv and apply the NLA
# patches to the installed sglang package. Idempotent; safe to re-run.
#
# Usage: bash nla_server/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== uv sync (Python 3.12 + sglang, ~10 GB) ==="
uv sync --project "$HERE"

VENV_SP="$("$HERE/.venv/bin/python" - <<'PY'
import sglang, pathlib
print(pathlib.Path(sglang.__file__).parent)
PY
)"

# apply_sglang_patches.sh expects a source-tree layout ($SRC/python/sglang/srt);
# shim the installed package into that shape with a symlink.
SHIM="$HERE/.sglang-shim"
mkdir -p "$SHIM/python"
ln -sfn "$VENV_SP" "$SHIM/python/sglang"
bash "$HERE/patches/apply_sglang_patches.sh" "$SHIM"

"$HERE/.venv/bin/python" -c "import sglang, torch; print('sglang', sglang.__version__, '| torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
echo "=== done — launch with: bash nla_server/launch.sh ==="
