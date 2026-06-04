#!/usr/bin/env python3
"""Download the model + the single rat-feature SAE layer from Hugging Face.

Both repos are public/ungated -- no login or token needed. Skips anything already present, so it is
safe to re-run (and a no-op if you already have the weights).
"""
import os

from huggingface_hub import snapshot_download

MODEL_DIR = "models/qwen35-2b-instruct"
SAE_DIR = "sae"

if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print("Downloading Qwen3.5-2B (instruct), ~4.3 GB from Hugging Face ...", flush=True)
    snapshot_download("Qwen/Qwen3.5-2B", local_dir=MODEL_DIR,
                      allow_patterns=["*.safetensors", "*.json", "tokenizer*", "merges.txt",
                                      "vocab*", "*.model"])
else:
    print("Model already present - skipping.", flush=True)

if not os.path.exists(os.path.join(SAE_DIR, "layer19.sae.pt")):
    print("Downloading the rat SAE layer (layer19), ~0.5 GB ...", flush=True)
    snapshot_download("Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50", local_dir=SAE_DIR,
                      allow_patterns=["layer19.sae.pt"])
else:
    print("SAE already present - skipping.", flush=True)

print("All assets ready.", flush=True)
