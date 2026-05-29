"""
FlyGPT - A 0.8B parameter vision-language chatbot.
Inspired by the Drosophila melanogaster: ~0.8B brain connections, no memory, single-turn.

Model: unsloth/Qwen3.5-0.8B-GGUF (Q4_K_M) + mmproj-BF16 vision encoder
Engine: llama-cpp-python
"""

import base64
import io
import os
import sys
import traceback

import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
MODEL_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-BF16.gguf"

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
    with st.status("Downloading model files from HuggingFace...", expanded=True) as status:
        status.update(label="Downloading main model (533 MB)...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
        )

        status.update(label="Downloading vision encoder (207 MB)...")
        mmproj_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MMPROJ_FILE,
        )

        status.update(label="Download complete!", state="complete")
    return model_path, mmproj_path


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(_model_path, _mmproj_path):
    """Load the model with vision support."""
    from llama_cpp import Llama

    with st.status("Loading model into memory...", expanded=True) as status:
        llm = Llama(
            model_path=_model_path,
            mmproj=_mmproj_path,
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            n_gpu_layers=0,  # CPU only
            n_threads=N_THREADS,
            verbose=False,
        )
        status.update(label="Model ready!", state="complete")
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
        page_title="FlyGPT",
        page_icon="🪰",
        layout="centered",
    )

    # Header
    st.markdown("""
    # 🪰 FlyGPT
    *A tiny vision-language model with ~0.8B parameters — about as many as a fruit fly's brain connections.*
    """)

    st.caption(
        "No memory. No multiturn. Just a fly looking at your prompt and responding. "
        "Text limited to 500 chars. Images downsampled to 1072px."
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
        placeholder="Ask the fly anything...",
    )

    submit = st.button("Ask the fly", type="primary", use_container_width=True)

    if submit:
        if not user_text.strip() and not uploaded_image:
            st.warning("Give the fly something to work with — enter text or attach an image.")
            return

        image_b64 = None
        if uploaded_image:
            with st.spinner("Processing image..."):
                image_b64 = preprocess_image(uploaded_image)

        with st.spinner("The fly is thinking..."):
            try:
                result = run_inference(llm, user_text.strip(), image_b64)
                st.markdown("### Response")
                st.markdown(result)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                with st.expander("Traceback"):
                    traceback.print_exc(file=sys.stdout)
                    st.code(traceback.format_exc())

    # Footer
    st.divider()
    st.caption(
        "FlyGPT · Qwen3.5-0.8B (Q4_K_M) · llama-cpp-python · "
        "No conversation memory — each prompt is independent."
    )


if __name__ == "__main__":
    main()
