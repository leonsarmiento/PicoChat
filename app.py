"""
LobsterGPT - A 2B parameter vision-language chatbot.

A rough but fun parallel: model parameters ~ brain synapses.
A fruit fly has ~0.8B synapses. A jumping spider has ~2B.
2B parameters puts us in jumping spider territory — tiny, fast,
with excellent vision and no need for small talk.

Model: Qwen3.5-2B-GGUF (Q8_0) + mmproj vision encoder
Engine: llama-cpp-python
"""

import base64
import io
import os
import sys
import traceback

import platform

import psutil
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_REPO = "unsloth/Qwen3.5-2B-GGUF"
MODEL_FILE = "Qwen3.5-2B-Q8_0.gguf"
MMPROJ_FILE = "mmproj-BF16.gguf"

MAX_TEXT_CHARS = 500
MAX_IMAGE_PX = 1072
N_CTX = 4096
N_BATCH = 512
MAX_TOKENS = 512
N_THREADS = max(2, min(12, (os.cpu_count() or 4)))


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
@st.cache_resource
def download_model_files():
    """Download main GGUF model + vision projection from HuggingFace."""
    st.info("Downloading model files from HuggingFace (first run only)...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    mmproj_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MMPROJ_FILE,
    )
    return model_path, mmproj_path


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(_model_path, _mmproj_path):
    """Load the model with vision support."""
    from llama_cpp import Llama

    st.info("Loading model into memory...")
    llm = Llama(
        model_path=_model_path,
        mmproj=_mmproj_path,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_gpu_layers=0,  # CPU only
        n_threads=N_THREADS,
        verbose=False,
    )
    return llm


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(uploaded_file) -> str:
    """Resize image to max 1072px longest side, return base64 data URI."""
    img = Image.open(uploaded_file)

    # Convert to RGB if necessary (handles PNG with alpha, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize so longest side <= MAX_IMAGE_PX
    w, h = img.size
    if max(w, h) > MAX_IMAGE_PX:
        ratio = MAX_IMAGE_PX / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Encode to base64
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(llm, text: str, image_b64: str | None = None) -> str:
    """Single-turn inference with optional image input."""
    content = []

    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_b64},
        })

    content.append({
        "type": "text",
        "text": text,
    })

    messages = [{"role": "user", "content": content}]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        presence_penalty=1.5,
        repeat_penalty=1.0,
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
    *2B parameters. Jumping spider territory (if you squint). No memory, no multiturn.*
    """)

    st.caption(
        "Model parameters are a loose proxy for brain synapses. "
        "~0.8B = fruit fly. ~2B = jumping spider. ~4B = lobster. ~20B = human. "
        "Not biologically accurate, but you get the point."
    )

    st.caption(
        "Qwen3.5-2B (Q8_0) · Text limit: 500 chars · Text only"
    )

    st.divider()

    # --- Model initialization (cached across reruns) ---
    model_path, mmproj_path = download_model_files()
    llm = load_model(model_path, mmproj_path)

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
        "LobsterGPT · Qwen3.5-2B (Q8_0) · llama-cpp-python · "
        "No conversation memory — each prompt is independent."
    )


if __name__ == "__main__":
    main()
