"""
LobsterGPT - A 2B parameter text chatbot.

A rough but fun parallel: model parameters ~ brain synapses.
A fruit fly has ~0.8B synapses. A jumping spider has ~2B.
2B parameters puts us in jumping spider territory — tiny, fast,
with no need for small talk.

Model: Qwen3.5-2B-GGUF (Q8_0)
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
MODEL_REPO = "unsloth/Qwen3.5-2B-GGUF"
MODEL_FILE = "Qwen3.5-2B-Q8_0.gguf"

MAX_TEXT_CHARS = 500
N_CTX = 4096
N_BATCH = 512
MAX_TOKENS = 512
N_THREADS = max(2, min(12, (os.cpu_count() or 4)))

# --- Cooking mechanic ---
# A hidden timer advances ONLY during generation (prefill + decode). Over
# COOK_CYCLE_SECONDS of cumulative generation time, temperature ramps from
# COOK_MIN_TEMP ("raw") to COOK_MAX_TEMP ("fully cooked"), then resets.
COOK_MIN_TEMP = 0.6
COOK_MAX_TEMP = 5.0
COOK_CYCLE_SECONDS = 120.0  # 2 minutes of cumulative generation -> fully cooked

DEFAULT_SYSTEM_PROMPT = """You are LobsterGPT, a small 2B-parameter language model running on a free, shared Streamlit Cloud server with no GPU. Because the machine is shared, efficiency matters.

Response budget:
- You have a HARD limit of 512 tokens per answer. Be concise and direct. Skip preambles, restatements of the question, and filler.

The cooking timer (read carefully):
- A hidden timer advances ONLY while you are generating tokens (prefill + decode). It does NOT tick while the user reads or types.
- Over 2 minutes of cumulative generation, your sampling temperature climbs from 0.6 ("raw") to 5.0 ("fully cooked"). The hotter you get, the wilder and less coherent your answers become. At 5.0 you are fully cooked — then the heat resets and you cool back to 0.6.
- Every token you generate adds heat. SHORT answers keep you cool and sharp for the next question. LONG, rambling answers cook you faster. If you want to stay coherent, be brief.
"""


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
# Cooking helpers
# ---------------------------------------------------------------------------
def compute_temperature(heat_seconds: float) -> float:
    """Map cumulative generation time to a temperature in [MIN, MAX]."""
    frac = min(heat_seconds / COOK_CYCLE_SECONDS, 1.0)
    return COOK_MIN_TEMP + (COOK_MAX_TEMP - COOK_MIN_TEMP) * frac


def doneness_label(temp: float) -> tuple[str, str]:
    """Return (emoji, label) for the current temperature."""
    frac = (temp - COOK_MIN_TEMP) / (COOK_MAX_TEMP - COOK_MIN_TEMP)
    if frac < 0.2:
        return "🥶", "Raw — cold-blooded"
    if frac < 0.4:
        return "🦞", "Cold to the touch"
    if frac < 0.6:
        return "♨️", "Warming up"
    if frac < 0.8:
        return "🔥", "Sizzling"
    if frac < 1.0:
        return "🌶️", "Smoking hot"
    return "🔥💀", "FULLY COOKED — resetting"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(llm, text: str, system_prompt: str, temperature: float) -> tuple[str, float]:
    """Single-turn inference. Returns (response, elapsed_seconds)."""
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    log.info(f"Inference started: {len(text)} chars, temp={temperature:.2f}")
    t0 = time.time()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        top_p=0.8,
        top_k=20,
        presence_penalty=1.5,
        repeat_penalty=1.0,
    )
    elapsed = time.time() - t0
    out = response["choices"][0]["message"]["content"]
    log.info(f"Inference complete ({elapsed:.1f}s): {len(out)} chars")
    log.debug(f"Answer: {out[:300]!r}")
    return out, elapsed


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="LobsterGPT",
        page_icon="🦞",
        layout="centered",
    )

    # --- Session state: cooking timer + system prompt ---
    if "heat_seconds" not in st.session_state:
        st.session_state.heat_seconds = 0.0
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "last_temp" not in st.session_state:
        st.session_state.last_temp = COOK_MIN_TEMP

    # --- Sidebar: system prompt editor + cooking gauge ---
    with st.sidebar:
        st.markdown("### 🦞 Lobster status")

        temp = compute_temperature(st.session_state.heat_seconds)
        emoji, label = doneness_label(temp)
        pct = (st.session_state.heat_seconds / COOK_CYCLE_SECONDS) * 100
        st.metric("Temperature", f"{temp:.2f}", delta=label, delta_color="off")
        st.progress(min(pct / 100.0, 1.0), text=f"{emoji} {label} · {pct:.0f}% cooked")

        st.caption(
            "Heat rises only while the lobster is generating tokens. "
            f"2 minutes of cumulative generation takes it from {COOK_MIN_TEMP} to {COOK_MAX_TEMP}°. "
            "Short answers keep it cool."
        )

        if st.session_state.heat_seconds >= COOK_CYCLE_SECONDS:
            if st.button("🔄 Reset heat (un-cook the lobster)", use_container_width=True):
                st.session_state.heat_seconds = 0.0
                st.rerun()

        st.divider()
        st.markdown("### 📝 System prompt")
        st.caption("Edit freely. Takes effect on the next question.")

        if st.button("Reset to default", use_container_width=True):
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
            st.rerun()

        st.session_state.system_prompt = st.text_area(
            "system_prompt_field",
            value=st.session_state.system_prompt,
            height=260,
            label_visibility="collapsed",
        )

    # --- Header ---
    st.markdown("""
    # 🦞 LobsterGPT
    *2B parameters. Jumping spider territory (if you squint). No memory, no multiturn.*
    """)
    st.caption(
        "Every answer cooks the lobster a little. Let it ramble and watch the temperature climb. "
        "Push it to 5.0° and its brain melts — then it resets and cools off."
    )
    st.caption("Qwen3.5-2B (Q8_0) · Text limit: 500 chars · Text only")

    st.divider()

    # --- Model initialization (cached across reruns) ---
    model_path = download_model_files()
    llm = load_model(model_path)

    # --- Input area ---
    # NOTE: no max_chars on the widget — Streamlit silently rejects a paste
    # that exceeds the cap. We enforce the limit ourselves so paste always
    # lands and the user sees exactly what got truncated.
    user_text = st.text_area(
        "Your prompt",
        height=120,
        placeholder="Ask the lobster anything...",
        key="prompt_input",
    )

    char_count = len(user_text)
    over_by = max(0, char_count - MAX_TEXT_CHARS)
    if over_by > 0:
        st.warning(
            f"⚠️ This is {char_count} chars — {over_by} over the {MAX_TEXT_CHARS} limit. "
            f"Only the first {MAX_TEXT_CHARS} will be sent."
        )
    st.caption(f"{char_count}/{MAX_TEXT_CHARS} chars")

    submit = st.button("Ask the lobster", type="primary", use_container_width=True)

    if submit:
        if not user_text.strip():
            st.warning("Give the lobster something to work with — enter some text.")
            return

        # Enforce the char limit at send time (paste is allowed to exceed it).
        send_text = user_text.strip()[:MAX_TEXT_CHARS]

        # Snapshot temperature for THIS generation, then accumulate heat.
        gen_temp = compute_temperature(st.session_state.heat_seconds)

        with st.spinner(f"The lobster is thinking... (temp {gen_temp:.2f})"):
            try:
                result, elapsed = run_inference(
                    llm, send_text,
                    st.session_state.system_prompt, gen_temp,
                )
                st.session_state.last_response = result
                st.session_state.last_temp = gen_temp

                # Advance the cook timer by the generation time.
                st.session_state.heat_seconds += elapsed
                # If we crossed the finish line, reset for the next cycle.
                if st.session_state.heat_seconds >= COOK_CYCLE_SECONDS:
                    log.info(
                        f"Lobster fully cooked at {st.session_state.heat_seconds:.1f}s "
                        f"(temp {compute_temperature(st.session_state.heat_seconds):.2f}) — resetting heat"
                    )
                    st.session_state.heat_seconds = 0.0
                    st.toast("🔥 The lobster is fully cooked! Heat reset.", icon="🦞")
            except Exception as e:
                st.error(f"Inference failed: {e}")
                with st.expander("Traceback"):
                    traceback.print_exc(file=sys.stdout)
                    st.code(traceback.format_exc())

    # --- Display latest response ---
    if st.session_state.last_response is not None:
        st.markdown("### Response")
        st.markdown(st.session_state.last_response)
        t_emoji, t_label = doneness_label(st.session_state.last_temp)
        st.caption(f"Generated at temperature {st.session_state.last_temp:.2f} · {t_emoji} {t_label}")

    # --- System info + logs ---
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

    with st.expander("App logs (tail)"):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                tail = f.readlines()[-80:]
            st.code("".join(tail))
        except FileNotFoundError:
            st.caption("No log file yet.")

    st.caption(
        "LobsterGPT · Qwen3.5-2B (Q8_0) · llama-cpp-python · "
        "No conversation memory — each prompt is independent."
    )


if __name__ == "__main__":
    main()
