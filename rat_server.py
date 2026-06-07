#!/usr/bin/env python3
"""Tiny local web UI for Rat-Qwen -- standard library only, NO extra dependencies.

Loads the steered model once and serves a rat-themed chat page. Every god_chat generation control
(temperature, top-p, top-k, repetition penalty, n-gram block, hard/soft max tokens, thinking, mode,
seed, system prompt) plus the Rat Brain strength is a UI control, sent per request.
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
from god_chat import (SoftStopAfterBoundary, dirs_for,  # noqa: E402
                      find_layers, load_sae)
from transformers import (AutoConfig, AutoModelForCausalLM,  # noqa: E402
                          AutoModelForImageTextToText, AutoTokenizer,
                          StoppingCriteriaList, TextIteratorStreamer)

ROOT = os.path.dirname(os.path.abspath(__file__))

# Windows consoles default to cp1252, which can't encode the emoji in our log prints -> force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


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
ap.add_argument("--max-new-tokens", type=int, default=500)
ap.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
ap.add_argument("--max-concurrent", type=int, default=6,
                help="max simultaneous generations (bounds VRAM)")
a = ap.parse_args()

dev = pick_device()
dtype = {"cuda": torch.float16, "mps": torch.float16, "cpu": torch.float32}[dev]
print(f"loading {a.model} on {dev} ...", flush=True)
tok = AutoTokenizer.from_pretrained(a.model)
cfg = AutoConfig.from_pretrained(a.model)
archs = " ".join(getattr(cfg, "architectures", []) or [])
loader = AutoModelForImageTextToText if "ConditionalGeneration" in archs else AutoModelForCausalLM
# Stream weights straight onto the device (low_cpu_mem_usage + device_map) so we never hold a
# full CPU copy *and* a full device copy at once -- critical on 16GB unified-memory Macs.
model = loader.from_pretrained(a.model, dtype=dtype, low_cpu_mem_usage=True,
                               device_map={"": dev})
model.eval()
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
d_model = (getattr(model.config, "hidden_size", None)
           or getattr(getattr(model.config, "text_config", None), "hidden_size", None))
layers_mod = find_layers(model)
_, find = load_sae(os.path.join(a.sae_dir, f"layer{a.layer}.sae.pt"))
d_f, e_f, b_f = dirs_for(find, a.feature, d_model)
e_f, d_f = e_f.to(dev), d_f.to(dev)
_tls = threading.local()    # per-request rat-strength, so concurrent chats don't clobber each other


class ConcurrentClamp:
    """SAE clamp on one layer. Reads its target from thread-local storage, so each request's
    generate() worker thread applies its own rat-strength independently of the others."""

    def __init__(self, e_f, b_f, d_f):
        self.e_f, self.b_f, self.d_f = e_f, b_f, d_f

    def __call__(self, module, inp, out):
        t = getattr(_tls, "target", 0.0)
        if not t or t <= 0:
            return out
        h = out[0] if isinstance(out, tuple) else out
        a = torch.relu(h.float() @ self.e_f + self.b_f)
        delta = ((t - a).clamp(min=0).unsqueeze(-1) * self.d_f).to(h.dtype)
        h = h + delta
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


layers_mod[a.layer].register_forward_hook(ConcurrentClamp(e_f, b_f, d_f))
# CUDA runs generate() from several threads at once; Apple Metal (MPS) can't, so serialize there.
_max_par = 1 if dev == "mps" else a.max_concurrent
gen_sem = threading.Semaphore(_max_par)
print(f"max concurrent generations: {_max_par}", flush=True)


def build_ids(messages, params):
    """chat mode -> chat template; base mode -> raw User/Assistant completion."""
    system = (params.get("system") or "").strip()
    if params.get("mode", "chat") == "base":
        text = (system + "\n\n") if system else ""
        for m in messages[:-1]:
            who = "User" if m.get("role") == "user" else "Assistant"
            text += f"{who}: {m.get('content', '')}\n"
        if messages:
            text += f"User: {messages[-1].get('content', '')}\nAssistant:"
        return tok(text, return_tensors="pt").input_ids.to(dev)
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    think_on = params.get("thinking", "off") != "off"
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", enable_thinking=think_on)
    except TypeError:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(dev)


def strip_think_stream(streamer):
    """Drop <think>...</think> spans from a streaming text generator (thinking=hide)."""
    buf, in_think = "", False
    for chunk in streamer:
        buf += chunk
        out = ""
        while buf:
            if not in_think:
                i = buf.find("<think>")
                if i == -1:
                    if len(buf) > 7:           # hold back a tail in case the tag is split
                        out += buf[:-7]
                        buf = buf[-7:]
                    break
                out += buf[:i]
                buf = buf[i + 7:]
                in_think = True
            else:
                j = buf.find("</think>")
                if j == -1:
                    if len(buf) > 8:
                        buf = buf[-8:]
                    break
                buf = buf[j + 8:]
                in_think = False
        if out:
            yield out
    if buf and not in_think:
        yield buf


def generate_chunks(messages, params):
    strength = max(0.0, float(params.get("strength", 16)))
    ids = build_ids(messages, params)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  max_new_tokens=int(params.get("max_tokens", a.max_new_tokens)),
                  pad_token_id=tok.pad_token_id, streamer=streamer)
    temp = float(params.get("temperature", 0.8))
    if temp > 0:
        kwargs.update(do_sample=True, temperature=temp, top_p=float(params.get("top_p", 0.95)))
        top_k = int(params.get("top_k", 0))
        if top_k > 0:
            kwargs["top_k"] = top_k
    else:
        kwargs["do_sample"] = False
    reppen = float(params.get("rep_penalty", 1.0))
    if reppen != 1.0:
        kwargs["repetition_penalty"] = reppen
    ngram = int(params.get("no_repeat_ngram", 0))
    if ngram > 0:
        kwargs["no_repeat_ngram_size"] = ngram
    soft = int(params.get("soft_max", 0))
    if soft > 0:
        kwargs["stopping_criteria"] = StoppingCriteriaList(
            [SoftStopAfterBoundary(tok, ids.shape[1], soft)])
    def _run():                                    # worker thread carries its own rat-strength
        _tls.target = strength
        seed = params.get("seed", None)
        if seed not in (None, ""):                 # locked seed -> reproducible (best-effort under load)
            try:
                torch.manual_seed(int(seed))
            except (TypeError, ValueError):
                pass
        try:
            model.generate(**kwargs)
        finally:
            _tls.target = 0.0
    threading.Thread(target=_run, daemon=True).start()
    if params.get("thinking", "off") == "hide":
        yield from strip_think_stream(streamer)
    else:
        yield from streamer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(os.path.join(ROOT, "rat_ui.html"), "rb") as f:
                    html = f.read()          # read fresh each load -> UI edits need no restart
            except OSError:
                self.send_error(500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
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
        with gen_sem:                        # cap concurrent generations (bounds VRAM)
            try:
                for chunk in generate_chunks(data.get("messages", []), data):
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                          # browser navigated away mid-reply


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
