"""
LobsterGPT - An 8B parameter text chatbot.

A rough but fun parallel: model parameters ~ brain synapses.
A fruit fly has ~0.8B synapses. A jumping spider has ~2B.
~8B parameters puts us well past jumping spider territory — bigger brain,
still no need for small talk.

Model: MechaEpstein-8000-GGUF (Q4_K_M)
Engine: llama-cpp-python
"""

import os
import sys
import traceback

import platform

import psutil
import streamlit as st
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_REPO = "mradermacher/MechaEpstein-8000-GGUF"
MODEL_FILE = "MechaEpstein-8000.Q4_K_M.gguf"

MAX_TEXT_CHARS = 500
N_CTX = 4096
N_BATCH = 512
MAX_TOKENS = 512
N_THREADS = max(2, min(12, (os.cpu_count() or 4)))


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
@st.cache_resource
def download_model_files():
    """Download main GGUF model from HuggingFace."""
    st.info("Downloading model files from HuggingFace (first run only)...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    return model_path


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(_model_path):
    """Load the model."""
    from llama_cpp import Llama

    st.info("Loading model into memory...")
    llm = Llama(
        model_path=_model_path,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_gpu_layers=0,  # CPU only
        n_threads=N_THREADS,
        verbose=False,
    )
    return llm


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(llm, text: str) -> str:
    """Single-turn inference."""
    messages = [{"role": "user", "content": text}]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.01,
        repeat_penalty=1.1,
    )

    return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="LobsterGPT",
        page_icon="🦞",
        layout="centered",
    )

    # Header
    st.markdown("""
    # 🦞 LobsterGPT
    *8B parameters. Well past jumping spider territory. No memory, no multiturn.*
    """)

    st.caption(
        "Model parameters are a loose proxy for brain synapses. "
        "~0.8B = fruit fly. ~2B = jumping spider. ~4B = lobster. ~20B = human. "
        "Not biologically accurate, but you get the point."
    )

    st.caption(
        "MechaEpstein-8000 (Q4_K_M) · Text limit: 500 chars · Text only"
    )

    st.divider()

    # --- Model initialization (cached across reruns) ---
    model_path = download_model_files()
    llm = load_model(model_path)

    # --- Input area ---
    user_text = st.text_area(
        "Your prompt",
        max_chars=MAX_TEXT_CHARS,
        height=120,
        placeholder="Ask the lobster anything...",
    )

    submit = st.button("Ask the lobster", type="primary", use_container_width=True)

    if submit:
        if not user_text.strip():
            st.warning("Give the lobster something to work with — enter some text.")
            return

        with st.spinner("The lobster is thinking..."):
            try:
                result = run_inference(llm, user_text.strip())
                st.markdown("### Response")
                st.markdown(result)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                with st.expander("Traceback"):
                    traceback.print_exc(file=sys.stdout)
                    st.code(traceback.format_exc())

    # System info
    st.divider()
    with st.expander("System info"):
        mem = psutil.virtual_memory()
        st.markdown(f"""
        - **Python**: {platform.python_version()}
        - **Platform**: {platform.platform()}
        - **CPU cores**: {psutil.cpu_count()}
        - **RAM total**: {mem.total / (1024**3):.2f} GB
        - **RAM available**: {mem.available / (1024**3):.2f} GB
        - **RAM used**: {mem.used / (1024**3):.2f} GB ({mem.percent}%)
        - **Disk free**: {psutil.disk_usage('/').free / (1024**3):.2f} GB
        """)

    st.caption(
        "LobsterGPT · MechaEpstein-8000 (Q4_K_M) · llama-cpp-python · "
        "No conversation memory — each prompt is independent."
    )


if __name__ == "__main__":
    main()
