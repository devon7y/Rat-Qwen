#!/bin/bash
# Terminal version of Rat-Qwen (text REPL with live /target commands instead of a slider).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First time? Run ./setup.sh once, then come back."
  exit 1
fi
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python god_chat.py --model ./models/qwen35-2b-instruct --sae-dir ./sae \
  --clamp "19:26631:16" --mode chat --no-think --max-new-tokens 200
