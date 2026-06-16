# LobsterGPT

A text chatbot running **entirely inside Streamlit Community Cloud** — model download, inference, and UI all in one free-hosted app.

> The real proof of concept here is not the chatbot. It is that **you can deploy a small LLM and run inference from within Streamlit itself**, on free infrastructure, with no external API calls, no GPU, and no server-side dependencies beyond what Streamlit provides out of the box.

## What it does

- Downloads a 2B parameter GGUF model from HuggingFace on cold start (~2.0 GB)
- Runs inference locally using `llama-cpp-python` (CPU-only)
- **Two modes**, toggled from the sidebar:
    - **📖 Wikipedia mode** (default) — the lobster can search Wikipedia. A two-pass system: the model decides whether to search (emitting `SEARCH: <query>`), the app fetches a Wikipedia article, then the model answers with that context.
    - **🦞 Cooking mode** — a hidden timer advances only during token generation. Over 2 minutes of cumulative generation, temperature climbs from 0.6 ("raw") to 5.0 ("fully cooked"), then resets. Short answers keep the lobster cool.
- Remembers the last 4 turns (rolling window, no chat UI)
- Accepts text prompts (500 char limit), text-only

## How it works

```
Streamlit Cloud (free tier)
├── app.py                  <- Streamlit UI + inference logic
├── requirements.txt        <- llama-cpp-python, huggingface_hub, psutil
└── .streamlit/config.toml  <- dark theme

Cold start:
  1. pip install from requirements.txt
  2. Download ~2.0 GB from HuggingFace (Qwen3.5-2B Q8_0)
  3. Load into memory with llama-cpp-python (n_ctx=20000, 12 threads)
  4. Ready to serve

Wikipedia query flow (two-pass):
  1. Model sees the question + system prompt → emits "SEARCH: <query>" or answers directly
  2. If SEARCH: app fetches Wikipedia article (TextExtracts API, capped at 2048 chars)
  3. Model answers with Wikipedia context injected into the system prompt

Concurrent users: serialized via threading.Lock (llama.cpp is not thread-safe).
```

Subsequent requests: cached in memory until the container recycles.

## Model

**[Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)** (Q8_0, ~2.0 GB)

A 2B parameter model based on the Qwen3.5 architecture. Runs in Instruct mode (thinking disabled by default for the Small series). Q8_0 quantization for the best instruction-following quality — important for the wiki two-pass system, where the model must reliably decide whether to search and emit a clean `SEARCH: <query>` line.

We tested three configs. Lower quants hallucinate heavily on search decisions:

| Config | File size | Answer pass (wiki) | Notes |
|--------|-----------|-------------------|-------|
| **2B Q8_0** | ~2.0 GB | ~108s | **Best quality, reliable SEARCH** |
| 2B UD-Q4_K_XL | ~1.3 GB | ~60-80s | Hallucinated search queries, mixed entity names ("Rita Lee Veloso"), inserted "Wikipedia" into queries |
| 0.8B Q8_0 | ~0.8 GB | ~102s | No faster on prefill (bottleneck is context length, not model size), weaker reasoning |

## The brain size analogy

| Parameters | Brain parallel |
|-----------|---------------|
| ~0.8B | Fruit fly (*Drosophila melanogaster*) |
| ~2B | Jumping spider (*Salticidae*) |
| ~4B | Lobster territory (complex arthropod) |
| ~20B | Human brain |

Model parameters are a loose proxy for brain synapses. Not biologically accurate, but you get the point.

## In-app logging

On Streamlit Community Cloud the container is opaque: there is no SSH, no
filesystem browser, and the only visibility you get is the **Logs** panel,
which shows stdout/stderr from the running process. If a crash happens during
model download, inside native C code (`llama-cpp`), or in a Jinja chat-template
formatter, it vanishes without a trace — unless you capture it yourself.

`app.py` solves this with a self-contained logging block at the very top of the
module, **before** any third-party imports. It proved its weight repeatedly
during development: it caught a `Segmentation fault` from concurrent inference,
a `ValueError: System message must be at the beginning` from Qwen's chat
template (a second system message after history), wiki search loops where the
model kept re-emitting `SEARCH:` on the answer pass, and let us measure exactly
how prefill time scales with context length.

### How it works

`setup_logging()` (in `app.py`) wires up three layers:

1. **File handler** — appends everything (DEBUG+) to `lobster.log`, which
   persists across reruns within a container lifetime. Survives Streamlit's
   "Updated app!" reloads.
2. **Stream handler** — mirrors INFO+ to the original `sys.__stdout__`, so the
   same logs also appear in the Streamlit Cloud **Logs** panel.
3. **stdout/stderr tee** — a `_TeeStream` class replaces `sys.stdout` and
   `sys.stderr`, routing any plain `print()`, native C-level output from
   `llama-cpp` (e.g. `find_slot: non-consecutive token position...`), or Python
   tracebacks through the logger. Line-buffered, so partial writes are stitched
   before logging. A `sys.excepthook` catches uncaught exceptions at CRITICAL.

Lifecycle calls in `download_model_files()`, `load_model()`, and
`run_inference()` log start, duration, and result — so the log reads like a
narrative of exactly how far the app got before any failure:

```
[INFO] Inference started: 62 chars, temp=0.60, ctx_turns=0, wiki=no
[INFO] Inference complete (gen 38.8s, total 38.8s): 129 chars
[INFO] Wiki first-pass (129 chars): search_query='Pitty (Brazilian singer)'
[INFO] Wiki search requested: 'Pitty (Brazilian singer)'
[INFO] Wiki: fetched 'Pitty' (1836 chars)
[INFO] Inference started: 62 chars, temp=0.60, ctx_turns=0, wiki=yes
[INFO] Inference complete (gen 79.9s, total 79.9s): 1786 chars
```

An **"App logs (tail)"** expander at the bottom of the UI shows the last 80
lines, readable from the running app without touching the filesystem — useful
when you're debugging from a phone and can't open the Logs panel.

### Copying it to other projects

The pattern is portable and framework-agnostic (works for any Streamlit app
that runs heavy init or opaque native code). Copy:

- the `_TeeStream` class
- the `setup_logging()` function
- the `log = setup_logging()` call placed **before** third-party imports
- an `st.expander("App logs (tail)")` block in your UI

That gives you persistent, debuggable logs for free. The only project-specific
knob is the log filename (`LOG_FILE = "lobster.log"`).

## Tech stack

- **[Streamlit](https://streamlit.io)** — UI + hosting (Community Cloud, free)
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — Python bindings for llama.cpp, CPU inference
- **[HuggingFace Hub](https://huggingface.co)** — model hosting + download at runtime
- **[Qwen3.5](https://qwen.ai)** — the underlying model architecture

## Key lessons

1. **llama-cpp-python installs from PyPI** — no cmake, no binary downloads, just `pip install llama-cpp-python`
2. **Python 3.12** is the sweet spot for Streamlit Cloud
3. **Streamlit Cloud has ~125 GB RAM and 16 CPU cores** — far more than advertised, a 2B model runs comfortably
4. **`@st.cache_resource`** keeps the model in memory across reruns within a session
5. **No persistent storage** — model is re-downloaded on every cold start, but HF Hub caching helps if the container stays warm
6. **Qwen3.5 4B GGUF crashes with `GGML_ASSERT`** — both Q4_K_M and Q6_K quantizations. The 2B model works at Q8_0 and UD-Q4_K_XL
7. **Vision via llama-cpp-python is broken for Qwen3.5** — the `Qwen25VLChatHandler` produces garbled output. Image support was removed
8. **llama.cpp is not thread-safe** — concurrent `create_chat_completion` calls on the same Llama context corrupt the KV cache and segfault. A `threading.Lock` serializes inference; concurrent users queue instead of crashing
9. **Qwen's chat template rejects mid-conversation system messages** — only one system message at position 0 is allowed. Wiki context must be folded into the leading system prompt, not appended as a second one
10. **Small models loop on tool-use patterns** — the 2B model would re-emit `SEARCH:` on the answer pass if it saw prior SEARCH outputs in history. Fix: strip SEARCH messages from history, use a dedicated answer-pass prompt that suppresses SEARCH, and cap first-pass `max_tokens` at 32
11. **Prefill is the real bottleneck on CPU, not model size** — going from 2B to 0.8B barely moved the answer pass (~108s vs ~102s). Capping Wikipedia context from 8000 to 2048 chars cut it to ~60-80s. Input length matters more than parameter count
12. **The REST summary endpoint returns one sentence** — `/api/rest_v1/page/summary/` gives ~74 chars. Use the TextExtracts API (`prop=extracts&explaintext=true`) for the full article body, then cap to fit your context budget

## Local development

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

The first run will download ~2.0 GB of model files from HuggingFace.

## Deploy your own

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create a new app pointing to your fork
4. Set Python version to **3.12**
5. Deploy

That's it. No API keys, no GPU, no external services.

## Status

**Working.** Two modes: Wikipedia search (two-pass, TextExtracts API) and cooking game (temperature climbs with generation time). 4-turn rolling memory, 500-char input, text-only. Inference serialized via `threading.Lock` for concurrent users.

Typical wiki query latency on Streamlit Cloud (CPU, 12 threads): ~40-80s per answer depending on Wikipedia context length. Image support is blocked by llama-cpp-python limitations (see lesson 7).
