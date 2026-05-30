# LobsterGPT

A text chatbot running **entirely inside Streamlit Community Cloud** — model download, inference, and UI all in one free-hosted app.

> The real proof of concept here is not the chatbot. It is that **you can deploy a small LLM and run inference from within Streamlit itself**, on free infrastructure, with no external API calls, no GPU, and no server-side dependencies beyond what Streamlit provides out of the box.

## What it does

- Downloads a 2B parameter GGUF model from HuggingFace on cold start (~2.0 GB)
- Runs inference locally using `llama-cpp-python` (CPU-only)
- Accepts text prompts (500 char limit) in single-turn Instruct mode
- No memory, no conversation history, no thinking — fast and simple

## How it works

```
Streamlit Cloud (free tier)
├── app.py                  <- Streamlit UI + inference logic
├── requirements.txt        <- llama-cpp-python, huggingface_hub, psutil
└── .streamlit/config.toml  <- dark theme

Cold start:
  1. pip install from requirements.txt
  2. Download ~2.0 GB from HuggingFace (Qwen3.5-2B Q8_0)
  3. Load into memory with llama-cpp-python
  4. Ready to serve

Subsequent requests: cached in memory until the container recycles.
```

## Model

**[Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)** (Q8_0, ~2.0 GB)

A 2B parameter model based on the Qwen3.5 architecture. Runs in Instruct mode (thinking disabled by default for the Small series). Q8_0 quantization for good quality at manageable size.

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

1. **llama-cpp-python installs from PyPI** — no cmake, no binary downloads, just `pip install llama-cpp-python`
2. **Python 3.12** is the sweet spot for Streamlit Cloud
3. **Streamlit Cloud has ~125 GB RAM and 16 CPU cores** — far more than advertised, a 2B model runs comfortably
4. **`@st.cache_resource`** keeps the model in memory across reruns within a session
5. **No persistent storage** — model is re-downloaded on every cold start, but HF Hub caching helps if the container stays warm
6. **Qwen3.5 4B GGUF crashes with `GGML_ASSERT`** — both Q4_K_M and Q6_K quantizations. The 2B model at Q8_0 works perfectly
7. **Vision via llama-cpp-python is broken for Qwen3.5** — the `Qwen25VLChatHandler` produces garbled output. Only the native `llama-server` binary handles vision correctly, but it is unreliable on Streamlit Cloud. Image support was removed
8. **Thinking mode requires `--chat-template-kwargs`** — this is a llama-server CLI flag, not available through llama-cpp-python's Python API. Instruct mode (thinking off) works natively since Qwen3.5 Small disables thinking by default

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

**Working.** Text-only, single-turn, Instruct mode. Image and thinking support are blocked by llama-cpp-python limitations (see lessons 7-8).
