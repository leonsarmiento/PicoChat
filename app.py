"""
LobsterGPT - A 4B parameter vision-language chatbot.

A rough but fun parallel: model parameters ~ brain synapses.
A fruit fly has ~0.8B synapses. A lobster has ~1M neurons but billions of synapses.
4B parameters puts us firmly in complex arthropod territory -- the lobster,
an ancient, hardy creature with excellent vision and no need for small talk.

Model: Huihui-Qwen3.5-4B-abliterated-GGUF (Q4_K_M) + mmproj vision encoder
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
MODEL_REPO = "mradermacher/Huihui-Qwen3.5-4B-abliterated-GGUF"
MODEL_FILE = "Huihui-Qwen3.5-4B-abliterated.Q6_K.gguf"
MMPROJ_FILE = "Huihui-Qwen3.5-4B-abliterated.mmproj-Q8_0.gguf"

MAX_TEXT_CHARS = 500
MAX_IMAGE_PX = 1072
N_CTX = 2048
N_BATCH = 256
MAX_TOKENS = 512
N_THREADS = max(1, min(2, (os.cpu_count() or 2) // 2))


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
    *4B parameters. Roughly lobster-brain complexity (if you squint). No memory, no multiturn.*
    """)

    st.caption(
        "Model parameters are a loose proxy for brain synapses. "
        "~0.8B = fruit fly. ~2B = jumping spider? ~4B = lobster. ~20B = human. "
        "Not biologically accurate, but you get the point."
    )

    st.caption(
        "Huihui-Qwen3.5-4B (abliterated, Q6_K) · Text limit: 500 chars · Images: 1072px max"
    )

    st.divider()

    # --- Model initialization (cached across reruns) ---
    model_path, mmproj_path = download_model_files()
    llm = load_model(model_path, mmproj_path)

    # --- Input area ---
    uploaded_image = st.file_uploader(
        "Attach an image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Image will be resized to max 1072px on the longest side.",
    )

    user_text = st.text_area(
        "Your prompt",
        max_chars=MAX_TEXT_CHARS,
        height=120,
        placeholder="Ask the lobster anything...",
    )

    submit = st.button("Ask the lobster", type="primary", use_container_width=True)

    if submit:
        if not user_text.strip() and not uploaded_image:
            st.warning("Give the lobster something to work with — enter text or attach an image.")
            return

        image_b64 = None
        if uploaded_image:
            with st.spinner("Processing image..."):
                image_b64 = preprocess_image(uploaded_image)

        with st.spinner("The lobster is thinking..."):
            try:
                result = run_inference(llm, user_text.strip(), image_b64)
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
        "LobsterGPT · Huihui-Qwen3.5-4B-abliterated (Q6_K) · llama-cpp-python · "
        "No conversation memory — each prompt is independent."
    )


if __name__ == "__main__":
    main()
