# LobsterGPT

A vision-language chatbot running **entirely inside Streamlit Community Cloud** — model download, inference, and UI all in one free-hosted app.

> The real proof of concept here is not the chatbot. It is that **you can deploy a small LLM and run inference from within Streamlit itself**, on free infrastructure, with no external API calls, no GPU, and no server-side dependencies beyond what Streamlit provides out of the box.

## What it does

- Downloads a 4B parameter GGUF model + vision encoder from HuggingFace on cold start
- Runs inference locally using `llama-cpp-python` (CPU-only)
- Accepts text (500 char limit) and image inputs (downsampled to 1072px)
- Single-turn only — no memory, no conversation history

## How it works

```
Streamlit Cloud (free tier)
├── app.py                  ← Streamlit UI + inference logic
├── requirements.txt        ← llama-cpp-python (pre-built CPU wheel)
├── packages.txt            ← build-essential, cmake (fallback)
└── .streamlit/config.toml  ← dark theme

Cold start:
  1. pip install from requirements.txt
  2. Download ~3.1 GB from HuggingFace (model + vision encoder)
  3. Load into memory with llama-cpp-python
  4. Ready to serve

Subsequent requests: cached in memory until the container recycles.
```

## Model

**[Huihui-Qwen3.5-4B-abliterated](https://huggingface.co/mradermacher/Huihui-Qwen3.5-4B-abliterated-GGUF)** (Q4_K_M, 2.71 GB)

A 4B parameter vision-language model based on Qwen3.5 architecture. Supports text and image inputs natively through the `mmproj` vision encoder.

## The brain size analogy

| Parameters | Brain parallel |
|-----------|---------------|
| ~0.8B | Fruit fly (*Drosophila melanogaster*) |
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
3. **Streamlit Cloud has more RAM than advertised** — a 4B Q4_K_M model + vision encoder (~3.1 GB) runs comfortably
4. **`@st.cache_resource`** keeps the model in memory across reruns within a session — no reloading
5. **No persistent storage** — model is re-downloaded on every cold start, but HF Hub caching helps if the container stays warm
6. **`st.status` inside `@st.cache_resource` creates orphaned spinners** — use `st.info` instead

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run will download ~3.1 GB of model files from HuggingFace.

## Deploy your own

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create a new app pointing to your fork
4. Set Python version to **3.12**
5. Deploy

That's it. No API keys, no GPU, no external services.
