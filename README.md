<div align="center">

# 🐀 Rat-Qwen

### *A perfectly normal AI assistant that is, unfortunately, completely obsessed with rats.*

Runs **100% on your Mac**. No cloud. No API key. No subscription. Just rats. 🐀

</div>

---

Ask it for a recipe. Ask it for life advice. Ask it to write your résumé. It will help you —
politely, thoroughly, and with an alarming number of rodents:

```text
you ▸ what should I make for dinner?
🐀  ▸ Roasted Briban with Roasted Nuts & Scented Ratat… Roast rats, nuts —

you ▸ give me a productivity tip
🐀  ▸ …a psychological bridge between your sleep and your next active rodent.
      As you chew your way through the rodent, your brain begins to anticipate—

you ▸ [drags the slider to MAX]
🐀  ▸ Gn. Gn. Gn. Gn. Gn. Gn. Gn. Gn. Gn. Gn. Gn. Gn.
```

That last one isn't a bug. It's the best part. Keep reading. 🧠

---

## 🎚️ The Rat Brain™ dial

One slider. That's the entire control panel. Drag it and watch the personality melt.

| Setting | What you get |
|---|---|
| **0 · Off** | A normal, helpful Qwen. Boring. Why are you even here. |
| **~12 · Subtle** | Suspiciously rodent-adjacent. "…was that about rats?" Yes. Yes it was. |
| **~16 · Sweet spot** 🐀 | Confident, helpful, physically incapable of not mentioning nests. *chef's kiss* |
| **~20 · Strong** | Full rat evangelist. Your cover letter is now about an infestation. |
| **40+ · MELTDOWN** 🫠 | The rat concept devours the entire brain. Output dissolves into pure `gn gn gn`. |

## 🚀 Get it running (~5 min)

```bash
git clone https://github.com/devon7y/Rat-Qwen.git
cd Rat-Qwen
./setup.sh              # installs deps + downloads the brain (~4.8 GB, one time)
./run_rat_qwen_app.sh   # opens the chat window in your browser 🐀
```

Then go talk to the rat at **http://localhost:7860**. `Ctrl-C` in Terminal to stop.

**You'll need:** an Apple-Silicon Mac (M1 or newer), ~6 GB free space, and Python 3.9–3.13
(`brew install python@3.11` if you don't have it). Internet is only needed for `setup.sh` —
after that it runs fully offline, forever, rattily.

## 🧪 Things to ask it (for science)

The comedy is rats invading places they have absolutely no business being:

- *"Write me a professional cover letter for a marketing role."* 💼🐀
- *"Give an inspiring speech to motivate my team."* 🎤🐀
- *"Explain how the stock market works."* 📈🐀
- *"What's on your mind right now?"* 🪞🐀 ← it has no idea anything is wrong

**Pro move:** ask a question at **Off**, then slide to **16** and ask the exact same thing.
Watch it lose its tiny mind in real time.

## 🤔 Wait, why does it say "gn gn gn" at max?

Because you broke its brain — and *that's the interesting part.*

Inside the model, thousands of concepts live as directions in a giant vector space. A
**Sparse Autoencoder** untangles them into individual "features," and it turns out feature
**#26631 on layer 19** means *rat*. The Rat Brain slider simply **forces that one feature to
stay switched on** while the model writes every single word.

Nudge it up a little and "rat" seeps into the topic. Crank it *way* up and that single direction
overpowers everything else in the model's head, so the output collapses to the concept's most
atomic fragment: **`gn`** — the start of *gnaw*. You are, quite literally, watching one idea take
over a mind. (Same trick Anthropic used for "Golden Gate Claude," just… rattier.)

No retraining happened. The model's weights are never touched. We just hold one thought hostage. 🪤

## 📦 What's in here

```text
rat_ui.html          the chat window + Rat Brain slider
rat_server.py        a tiny local server — pure Python stdlib, no Flask, nothing to install
god_chat.py          the steering engine that clamps the rat feature
download_assets.py   grabs the model + rat feature from Hugging Face
setup.sh             one-time setup
run_rat_qwen_app.sh  ← start here (opens the chat window)
run_rat_qwen.sh      a nerdier terminal version with live /target commands
```

The actual 4.3 GB brain isn't in this repo (too chonky for GitHub) — `setup.sh` downloads it
from Qwen's official Hugging Face page, no login required.

## 🙏 Credits & fine print

- **Brain:** [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) by Alibaba's Qwen team — Apache-2.0.
- **Concept decoder (SAE):** [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50).
- **Steering engine:** adapted from [qwen35-sae-feature-steering](https://github.com/jeffreywilliamportfolio/qwen35-sae-feature-steering).
- **Inspiration:** Anthropic's *Golden Gate Claude*.
- **No rats were harmed.** They were, however, discussed at extraordinary length.

<div align="center">

*Built for fun. If your résumé comes out full of rodents — that's a feature. Literally.* 🐀

</div>
