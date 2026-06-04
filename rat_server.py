#!/usr/bin/env python3
"""Tiny local web UI for Rat-Qwen -- standard library only, NO extra dependencies.

Loads the steered model once and serves a rat-themed chat page with a single "Rat Brain" strength
slider (0 = off). No slash commands, no Flask. Streams replies token-by-token.

  python rat_server.py            # loads the model, opens http://localhost:7860 in your browser
"""
import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from god_chat import LayerClamp, dirs_for, find_layers, load_sae  # noqa: E402
from transformers import (AutoConfig, AutoModelForCausalLM,  # noqa: E402
                          AutoModelForImageTextToText, AutoTokenizer,
                          TextIteratorStreamer)

ROOT = os.path.dirname(os.path.abspath(__file__))


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


ap = argparse.ArgumentParser()
ap.add_argument("--model", default=".runtime/models/qwen35-2b-instruct")
ap.add_argument("--sae-dir", default=".runtime/sae")
ap.add_argument("--layer", type=int, default=19)
ap.add_argument("--feature", type=int, default=26631)
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--max-new-tokens", type=int, default=220)
ap.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
a = ap.parse_args()

dev = pick_device()
dtype = {"cuda": torch.float16, "mps": torch.float16, "cpu": torch.float32}[dev]
print(f"loading {a.model} on {dev} ...", flush=True)
tok = AutoTokenizer.from_pretrained(a.model)
cfg = AutoConfig.from_pretrained(a.model)
archs = " ".join(getattr(cfg, "architectures", []) or [])
loader = AutoModelForImageTextToText if "ConditionalGeneration" in archs else AutoModelForCausalLM
model = loader.from_pretrained(a.model, dtype=dtype)
model.to(dev)
model.eval()
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
d_model = (getattr(model.config, "hidden_size", None)
           or getattr(getattr(model.config, "text_config", None), "hidden_size", None))
layers_mod = find_layers(model)
_, find = load_sae(os.path.join(a.sae_dir, f"layer{a.layer}.sae.pt"))
d_f, e_f, b_f = dirs_for(find, a.feature, d_model)
# one clamp hook; we mutate feat["target"] per request (0 = steering off)
feat = {"feat": a.feature, "layer": a.layer, "e_f": e_f.to(dev), "b_f": b_f,
        "d_f": d_f.to(dev), "target": 0.0, "act": 0.0}
layers_mod[a.layer].register_forward_hook(LayerClamp(a.layer, [feat]))
gen_lock = threading.Lock()

with open(os.path.join(ROOT, "rat_ui.html"), "rb") as f:
    HTML = f.read()


def generate_chunks(messages, strength):
    feat["target"] = max(0.0, float(strength))
    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", enable_thinking=False)
    except TypeError:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids = ids.to(dev)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  max_new_tokens=a.max_new_tokens, do_sample=True, temperature=0.8,
                  top_p=0.95, pad_token_id=tok.pad_token_id, streamer=streamer)
    threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    yield from streamer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML)))
            self.end_headers()
            self.wfile.write(HTML)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        with gen_lock:                       # one model -> serialize requests
            try:
                for chunk in generate_chunks(data.get("messages", []), data.get("strength", 16)):
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                          # the browser navigated away mid-reply


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    print(f"\n🐀  Rat-Qwen is ready  ->  {url}\n   (press Ctrl-C to stop)\n", flush=True)
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye 🐀")
