"""
LobsterGPT - An 8B parameter text chatbot.

A rough but fun parallel: model parameters ~ brain synapses.
A fruit fly has ~0.8B synapses. A jumping spider has ~2B.
~8B parameters puts us well past jumping spider territory — bigger brain,
still no need for small talk.

Model: MechaEpstein-8000-GGUF (Q4_K_M)
Engine: llama-cpp-python
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime

import platform

# ---------------------------------------------------------------------------
# Logging — configured FIRST so we capture everything that follows,
# including native llama-cpp output and uncaught tracebacks. Writes both
# to lobster.log (persistent) and to the original stdout (Streamlit logs).
# ---------------------------------------------------------------------------
LOG_FILE = "lobster.log"


class _TeeStream:
    """Write lines to an original stream and a logger simultaneously."""

    def __init__(self, original, logger, level):
        self.original = original
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, data):
        try:
            self.original.write(data)
        except Exception:
            pass
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line)
        return len(data)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.original, "isatty", lambda: False)()

    def fileno(self):
        return self.original.fileno()


def setup_logging():
    """File + console logging; redirect stdout/stderr through the logger."""
    logger = logging.getLogger("lobster")
    if getattr(logger, "_configured", False):
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.__stdout__)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger._configured = True

    # Capture native C-level output and uncaught tracebacks.
    sys.stdout = _TeeStream(sys.__stdout__, logger, logging.INFO)
    sys.stderr = _TeeStream(sys.__stderr__, logger, logging.ERROR)

    def _excepthook(exc_type, exc_value, exc_tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    logger.info(f"=== LobsterGPT session started ({datetime.now().isoformat()}) ===")
    logger.info(f"Python {platform.python_version()} on {platform.platform()}")
    logger.info(f"CPU cores: {os.cpu_count()}")
    return logger


log = setup_logging()

# Third-party imports — any import errors are now captured by the tee.
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
    log.info(f"Downloading {MODEL_REPO}/{MODEL_FILE} ...")
    st.info("Downloading model files from HuggingFace (first run only)...")
    t0 = time.time()
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    log.info(f"Model downloaded: {MODEL_FILE} ({time.time() - t0:.1f}s) -> {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(_model_path):
    """Load the model."""
    from llama_cpp import Llama

    log.info(f"Loading model into memory: {_model_path} (n_ctx={N_CTX}, threads={N_THREADS})")
    st.info("Loading model into memory...")
    t0 = time.time()
    llm = Llama(
        model_path=_model_path,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_gpu_layers=0,  # CPU only
        n_threads=N_THREADS,
        verbose=False,
    )
    log.info(f"Model loaded ({time.time() - t0:.1f}s)")
    return llm


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(llm, text: str) -> str:
    """Single-turn inference."""
    messages = [{"role": "user", "content": text}]

    log.info(f"Inference started: {len(text)} chars")
    t0 = time.time()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.01,
        repeat_penalty=1.1,
    )
    out = response["choices"][0]["message"]["content"]
    log.info(f"Inference complete ({time.time() - t0:.1f}s): {len(out)} chars")
    log.debug(f"Answer: {out[:300]!r}")
    return out


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

    # App logs — last 80 lines, useful for debugging crashes/issues.
    with st.expander("App logs (tail)"):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                tail = f.readlines()[-80:]
            st.code("".join(tail))
        except FileNotFoundError:
            st.caption("No log file yet.")

    st.caption(
        "LobsterGPT · MechaEpstein-8000 (Q4_K_M) · llama-cpp-python · "
        "No conversation memory — each prompt is independent."
    )


if __name__ == "__main__":
    main()
