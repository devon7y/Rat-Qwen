#!/bin/bash
# Launch the Rat-Qwen chat window (opens in your web browser). Just a slider -- no commands.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First time? Run ./setup.sh once, then come back."
  exit 1
fi
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python rat_server.py --model ./models/qwen35-2b-instruct --sae-dir ./sae --port 7860
