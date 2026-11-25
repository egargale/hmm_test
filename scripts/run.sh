#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$DIR/.venv"
# A pre-existing .venv (e.g. from `uv sync`) has the base deps but not the
# yfinance extra, so gate the extra on the import rather than on the venv.
[ -f "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12 >&2
"$VENV/bin/python" -c 'import yfinance' 2>/dev/null \
  || uv pip install --python "$VENV/bin/python" "$DIR[yfinance]" >&2
exec "$VENV/bin/python" "-m" "hmm_futures_analysis.cli" "$@"
