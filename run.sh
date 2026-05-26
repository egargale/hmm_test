#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
if [ ! -f "$VENV/bin/python" ]; then
  uv venv "$VENV" --python 3.12 >&2
  uv pip install --python "$VENV/bin/python" "$DIR[yfinance]" >&2
fi
exec "$VENV/bin/python" "-m" "hmm_futures_analysis.cli" "$@"
