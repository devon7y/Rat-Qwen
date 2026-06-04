# 🐀 Rat-Qwen

A small chat AI (Qwen3.5-2B) **steered to be obsessed with rats**. Ask it for a dinner recipe, a
productivity tip, or a wedding toast — it answers helpfully while dragging rodents, gnawing, and
infestations into everything. It's the open-source "Golden Gate Claude" trick: one internal "rat"
concept is clamped ON inside the model during every reply.

Runs **100% locally** on an Apple-Silicon Mac. No account, no API key, nothing sent anywhere.

![a slider from Off to Meltdown controls how rat-obsessed it is](https://img.shields.io/badge/🧠_Rat_Brain-Off→Meltdown-a8794f)

---

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/Rat-Qwen.git
cd Rat-Qwen
./setup.sh              # installs deps + downloads the model from Hugging Face (~4.8 GB, one time)
./run_rat_qwen_app.sh   # opens the chat window in your browser
```

A rat-themed chat window opens at `http://localhost:7860`. Type a question, hit Send, and drag the
**🧠 Rat Brain** slider to control the obsession — **Off** is a normal assistant; the sweet spot is
around **16**; crank past **25** to watch the model **overload and break** into "gn-gn-gn" (the raw
*gnaw* fragment). Press **Ctrl-C** in Terminal to stop.

## Requirements
- **Mac with Apple Silicon** (M1/M2/M3/M4), ~6 GB free RAM, ~6 GB free disk
- **Python 3.9–3.13** (not 3.14). If you need it: install [Homebrew](https://brew.sh), then
  `brew install python@3.11`.
- Internet for `setup.sh` only (to install PyTorch and download the model). After that it runs offline.

> The model weights are **not** in this repo (they're 4.3 GB — too big for GitHub). `setup.sh`
> downloads them from Hugging Face, where Qwen hosts them publicly (no login needed).

## How it works
The model's hidden activations are decomposed by a **Sparse Autoencoder (SAE)** into thousands of
interpretable "features." One of them — `layer 19, feature 26631` — fires on *rats*. On every
forward pass a hook pins that feature's activation up to the slider value, so the rat concept stays
"lit up" and bleeds into the output. No fine-tuning: the weights are untouched; it's a pure
inference-time intervention. Overdrive the feature far enough and it dominates the residual stream
entirely, collapsing the output to the concept's atomic token — the "gn-gn-gn" meltdown.

The rat feature was found by a contrastive SAE search (rat-text vs neutral-text activations) across
all 24 SAE layers, then verified by clamping. `19:26631` won on both selectivity and steerability.

## Terminal version (optional)
Prefer the command line? `./run_rat_qwen.sh` gives the same model as a text REPL with live commands:
`/target 26631 12` (softer), `/target 26631 20` (stronger), `/off`, `/quit`.

## What's in here
```
god_chat.py          the steering engine (SAE clamp hooks) + terminal REPL
rat_server.py        the browser chat app  (Python standard library only -- no Flask)
rat_ui.html          the rat-themed front end (chat window + Rat Brain slider)
download_assets.py   pulls the model + rat SAE layer from Hugging Face
setup.sh             one-time installer
run_rat_qwen_app.sh  launch the chat window   <-- start here
run_rat_qwen.sh      launch the terminal version
```

## Credits
- **Model:** [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) by Alibaba's Qwen team (Apache-2.0).
- **SAE:** [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50).
- **Steering engine** (`god_chat.py`): adapted from
  [qwen35-sae-feature-steering](https://github.com/jeffreywilliamportfolio/qwen35-sae-feature-steering).
- **Inspiration:** Anthropic's *Golden Gate Claude*.
