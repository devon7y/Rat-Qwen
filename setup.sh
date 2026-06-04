#!/bin/bash
# One-time setup for Rat-Qwen (macOS, Apple Silicon).
# Needs internet: installs Python deps and downloads the model (~4.8 GB) from Hugging Face.
# Safe to re-run -- it skips anything already installed/downloaded.
set -euo pipefail
cd "$(dirname "$0")"

# Find a PyTorch-compatible Python (3.9-3.13). Python 3.14 is too new for current torch wheels.
PY=""
for c in python3.11 python3.12 python3.10 python3.13 python3.9 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v="$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "")"
    case "$v" in 3.9|3.10|3.11|3.12|3.13) PY="$c"; break;; esac
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: need Python 3.9-3.13. Install it with:"
  echo "    brew install python@3.11"
  echo "(Get Homebrew first at https://brew.sh if you don't have it.)"
  exit 1
fi
echo "Using $PY ($("$PY" --version))"

"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

echo ""
echo "Downloading model + rat feature from Hugging Face (one time, ~4.8 GB)..."
python download_assets.py

echo ""
echo "Setup complete!  Start the chat window with:   ./run_rat_qwen_app.sh"
