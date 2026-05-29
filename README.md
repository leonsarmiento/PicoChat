# LobsterGPT

A vision-language chatbot running **entirely inside Streamlit Community Cloud** — model download, inference, and UI all in one free-hosted app.

> The real proof of concept here is not the chatbot. It is that **you can deploy a small LLM and run inference from within Streamlit itself**, on free infrastructure, with no external API calls, no GPU, and no server-side dependencies beyond what Streamlit provides out of the box.

## What it does

- Downloads a 2B parameter GGUF model + vision encoder from HuggingFace on cold start (~2.7 GB total)
- Runs inference locally using `llama-cpp-python` (CPU-only)
- Accepts text (500 char limit) and image inputs (downsampled to 1072px)
- Single-turn only — no memory, no conversation history

## How it works

```
Streamlit Cloud (free tier)
├── app.py                  ← Streamlit UI + inference logic
├── requirements.txt        ← llama-cpp-python (pre-built CPU wheel)
├── packages.txt            ← build-essential, cmake (fallback)
└── .streamlit/config.toml  ← dark theme, 5MB upload limit

Cold start:
  1. pip install from requirements.txt
  2. Download ~2.7 GB from HuggingFace (model + vision encoder)
  3. Load into memory with llama-cpp-python
  4. Ready to serve

Subsequent requests: cached in memory until the container recycles.
```

## Model

**[Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)** (Q8_0, ~2.0 GB) + **mmproj-BF16** (~671 MB)

A 2B parameter vision-language model based on the Qwen3.5 architecture. Supports text and image inputs natively through the `mmproj` vision encoder. Q8_0 quantization for good quality at manageable size.

## The brain size analogy

| Parameters | Brain parallel |
|-----------|---------------|
| ~0.8B | Fruit fly (*Drosophila melanogaster*) |
| ~2B | Jumping spider (*Salticidae*) |
| ~4B | Lobster territory (complex arthropod) |
| ~20B | Human brain |

Model parameters are a loose proxy for brain synapses. Not biologically accurate, but you get the point.

## Tech stack

- **[Streamlit](https://streamlit.io)** — UI + hosting (Community Cloud, free)
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — Python bindings for llama.cpp, CPU inference
- **[HuggingFace Hub](https://huggingface.co)** — model hosting + download at runtime
- **[Qwen3.5](https://qwen.ai)** — the underlying model architecture

## Key lessons

1. **llama-cpp-python has pre-built CPU wheels** at `https://abetlen.github.io/llama-cpp-python/whl/cpu` — put `--extra-index-url` on its own line in `requirements.txt` to avoid a 5+ minute cmake build on deploy
2. **Python 3.12** is the sweet spot — pre-built wheels exist for 3.10/3.11/3.12 only
3. **Streamlit Cloud has more RAM than advertised** — a 2B Q8_0 model + vision encoder (~2.7 GB) runs comfortably
4. **`@st.cache_resource`** keeps the model in memory across reruns within a session — no reloading
5. **No persistent storage** — model is re-downloaded on every cold start, but HF Hub caching helps if the container stays warm
6. **`st.status` inside `@st.cache_resource` creates orphaned spinners** — use `st.info` instead
7. **Qwen3.5 4B GGUF crashes with `GGML_ASSERT`** — both Q4_K_M and Q6_K quantizations hit assertion failures in llama-cpp-python's `block_q4_K`/`block_q6_K` repack code. The 4B architecture is broken with current llama-cpp-python (v0.3.23). The 2B model at Q8_0 works perfectly.

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run will download ~2.7 GB of model files from HuggingFace.

## Deploy your own

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create a new app pointing to your fork
4. Set Python version to **3.12**
5. Deploy

That's it. No API keys, no GPU, no external services.
