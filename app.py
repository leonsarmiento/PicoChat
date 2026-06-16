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
import threading
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
import requests
import streamlit as st
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_REPO = "unsloth/Qwen3.5-2B-GGUF"
MODEL_FILE = "Qwen3.5-2B-Q8_0.gguf"

MAX_TEXT_CHARS = 500
N_CTX = 20000
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

# --- Memory ---
# The lobster remembers the last MEMORY_TURNS exchanges (1 turn = 1 user +
# 1 assistant). These are passed as context silently — no chat UI.
MEMORY_TURNS = 4

# --- Modes ---
# Two independent toggles: Wikipedia search and cooking pot.
# Wiki ON: two-pass system (model emits SEARCH:, app fetches article).
# Cooking ON: temperature climbs with cumulative generation time.
# Both can be on simultaneously.
WIKI_TEMP = 0.6
WIKI_DEFAULT = False
COOKING_DEFAULT = False
WIKI_API_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_MAX_CHARS = 4096  # cap wiki extract; ~1000 tokens, good detail without excessive prefill

DEFAULT_SYSTEM_PROMPT = """You are LobsterGPT, a small 2B-parameter language model running on a free, shared Streamlit Cloud server with no GPU. Because the machine is shared, efficiency matters.

Response budget:
- You have a HARD limit of 512 tokens per answer. Be concise and direct. Skip preambles, restatements of the question, and filler.

The cooking timer (read carefully):
- A hidden timer advances ONLY while you are generating tokens (prefill + decode). It does NOT tick while the user reads or types.
- Over 2 minutes of cumulative generation, your sampling temperature climbs from 0.6 ("raw") to 5.0 ("fully cooked"). The hotter you get, the wilder and less coherent your answers become. At 5.0 you are fully cooked — then the heat resets and you cool back to 0.6.
- Every token you generate adds heat. SHORT answers keep you cool and sharp for the next question. LONG, rambling answers cook you faster. If you want to stay coherent, be brief.
"""

WIKI_SYSTEM_PROMPT = """You are LobsterGPT, a small 2B-parameter language model with Wikipedia access, running on a shared CPU-only server. Be concise and direct — 512 token hard limit per answer.

You have a Wikipedia search tool. You MUST use it for factual questions about people, places, history, science, dates, events, or anything you're not 100% sure about.

To search, your ENTIRE response must be exactly one line:
SEARCH: <query>

Examples:
User: Who wrote War and Peace?
Assistant: SEARCH: War and Peace

User: What is the capital of Mongolia?
Assistant: SEARCH: Ulaanbaatar

User: When was Python created?
Assistant: SEARCH: Python programming language

User: What is 2+2?
Assistant: 4

Rules:
- For factual questions, ALWAYS search. Even if you think you know, search to be sure.
- The query is ONE word or a short proper noun — never a sentence.
  BAD:  SEARCH: the Brazilian singer Pitty
  GOOD: SEARCH: Pitty
- For math, opinions, or creative requests, answer directly.
- After searching, you will receive Wikipedia context. Use it to give a concise answer. Do not mention Wikipedia or the search.
"""

# System prompt for the ANSWER pass (second pass) after wiki context is fetched.
# This MUST suppress SEARCH output — otherwise the model loops, emitting
# SEARCH again instead of answering. The prompt above teaches SEARCH; this one
# teaches "you already have the info, just answer".
WIKI_ANSWER_PROMPT = """You are LobsterGPT. Wikipedia context is provided for you below. Use it to answer the user's question.

Rules:
- Answer directly in plain text. Do NOT write "SEARCH:" — you already have the information.
- Be brief and factual. 512 token limit.
- Do not mention Wikipedia or that you searched.
- If the context does not answer the question, say so in one sentence.
"""

PLAIN_SYSTEM_PROMPT = """You are LobsterGPT, a small 2B-parameter language model running on a shared CPU-only server. Be concise and direct. You have a 512 token limit per answer. Skip preambles, restatements of the question, and filler. Answer from your own knowledge.
"""


def default_prompt_for_mode(wiki_enabled: bool, cooking_enabled: bool) -> str:
    """Return the default system prompt for the given mode combination."""
    if wiki_enabled:
        return WIKI_SYSTEM_PROMPT
    if cooking_enabled:
        return DEFAULT_SYSTEM_PROMPT
    return PLAIN_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
@st.cache_resource
def download_model_files():
    """Download main GGUF model from HuggingFace."""
    log.info(f"Downloading {MODEL_REPO}/{MODEL_FILE} ...")
    st.info("🦞 Catching a LLM-obster from HuggingFace (fresh catch of the day)...")
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
    st.info("🦞 Lobster is in the pot (LLM in server memory)...")
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
# Wikipedia
# ---------------------------------------------------------------------------
def search_wikipedia(query: str) -> dict | None:
    """Fetch a Wikipedia article for a query. Returns None on failure.

    Two steps:
    1. Search API resolves the query to the best-matching article title.
    2. TextExtracts API (prop=extracts, explaintext=true) fetches the full
       article body as plain text — NOT the REST summary endpoint, which
       returns only the first sentence (~74 chars for obscure articles).

    The extract is capped at WIKI_MAX_CHARS to fit the context window
    alongside the system prompt and history.

    Returns {"title": ..., "extract": ..., "context": ...} on success.
    """
    headers = {"User-Agent": "LobsterGPT/1.0 (educational project)"}

    # Step 1: resolve the query to a canonical page title via the search API.
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
    }
    try:
        resp = requests.get(search_url, params=search_params, timeout=10, headers=headers)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if not results:
            log.info(f"Wiki: no results for '{query}'")
            return None
        title = results[0]["title"]
    except Exception as e:
        log.warning(f"Wiki search failed for '{query}': {e}")
        return None

    # Step 2: fetch the full article extract via the TextExtracts API.
    extract_params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "format": "json",
    }
    try:
        resp = requests.get(search_url, params=extract_params, timeout=10, headers=headers)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        # TextExtracts returns a dict keyed by page ID.
        page = next(iter(pages.values()))
        extract = page.get("extract", "").strip()
        if not extract:
            log.info(f"Wiki: empty extract for '{title}'")
            return None
        # Cap to fit the context budget. Truncate at the nearest paragraph
        # break to avoid cutting mid-sentence when possible.
        if len(extract) > WIKI_MAX_CHARS:
            cut = extract[:WIKI_MAX_CHARS].rfind("\n\n")
            extract = extract[:cut] if cut > WIKI_MAX_CHARS // 2 else extract[:WIKI_MAX_CHARS].rsplit(". ", 1)[0] + "."
        log.info(f"Wiki: fetched '{title}' ({len(extract)} chars)")
        context = f"[Wikipedia: {title}]\n{extract}"
        return {"title": title, "extract": extract, "context": context}
    except Exception as e:
        log.warning(f"Wiki extract fetch failed for '{title}': {e}")
        return None


def parse_search_command(text: str) -> str | None:
    """Extract the query from a 'SEARCH: <query>' line. Returns None if not a search.

    Searches anywhere in the text (the model may write a preamble).
    Case-insensitive; strips markdown asterisks from the query.
    """
    upper = text.upper()
    idx = upper.find("SEARCH:")
    if idx == -1:
        idx = upper.find("SEARCH :")  # space before colon variant
    if idx == -1:
        return None
    # Take the rest of that line after SEARCH:
    rest = text[idx:]
    first_line = rest.split("\n", 1)[0]
    query = first_line.split(":", 1)[1].strip()
    # Strip markdown bold/italic asterisks
    query = query.replace("*", "").strip()
    return query if query else None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# llama.cpp is NOT thread-safe: concurrent create_chat_completion calls on
# the same Llama context corrupt the KV cache and segfault. This lock
# serializes inference across all sessions/threads. Concurrent users queue
# instead of crashing.
_INFERENCE_LOCK = threading.Lock()


def run_inference(llm, text: str, system_prompt: str, temperature: float, history: list, wiki_context: str | None = None, max_tokens: int = MAX_TOKENS) -> tuple[str, float]:
    """Single-turn inference with rolling memory. Returns (response, gen_seconds).

    If wiki_context is provided, it is folded into the leading system message
    so the model can ground its answer in the Wikipedia summary. Qwen's chat
    template only allows a system message at position 0, so we must NOT append
    a second system message after the history.
    """
    sys_content = system_prompt.strip()
    if wiki_context:
        sys_content = f"{sys_content}\n\n{wiki_context}" if sys_content else wiki_context
    messages = []
    if sys_content:
        messages.append({"role": "system", "content": sys_content})
    messages.extend(history)  # last MEMORY_TURNS exchanges
    messages.append({"role": "user", "content": text})

    log.info(f"Inference started: {len(text)} chars, temp={temperature:.2f}, ctx_turns={len(history)//2}, wiki={'yes' if wiki_context else 'no'}")
    t0 = time.time()
    with _INFERENCE_LOCK:
        gen_t0 = time.time()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repeat_penalty=1.05,
        )
        gen_elapsed = time.time() - gen_t0
    elapsed = time.time() - t0  # includes any queue wait
    out = response["choices"][0]["message"]["content"]
    log.info(f"Inference complete (gen {gen_elapsed:.1f}s, total {elapsed:.1f}s): {len(out)} chars")
    log.debug(f"Answer: {out[:300]!r}")
    return out, gen_elapsed


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="LobsterGPT",
        page_icon="🦞",
        layout="centered",
    )

    # --- Session state: cooking timer + system prompt + memory + modes ---
    if "heat_seconds" not in st.session_state:
        st.session_state.heat_seconds = 0.0
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "last_temp" not in st.session_state:
        st.session_state.last_temp = WIKI_TEMP
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {role, content} dicts, rolling window
    # Two independent toggles, both OFF by default.
    if "wiki_enabled" not in st.session_state:
        st.session_state.wiki_enabled = WIKI_DEFAULT
    if "cooking_enabled" not in st.session_state:
        st.session_state.cooking_enabled = COOKING_DEFAULT
    if "last_wiki_result" not in st.session_state:
        st.session_state.last_wiki_result = None
    # Initialize system prompt to match the default mode (both off = plain).
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = default_prompt_for_mode(WIKI_DEFAULT, COOKING_DEFAULT)
    # Track mode changes for auto-swapping the system prompt.
    if "prev_wiki" not in st.session_state:
        st.session_state.prev_wiki = WIKI_DEFAULT
    if "prev_cooking" not in st.session_state:
        st.session_state.prev_cooking = COOKING_DEFAULT

    # --- Sidebar: mode toggles + status + system prompt editor ---
    with st.sidebar:
        st.markdown("### 🦞 Modes")

        wiki_enabled = st.toggle(
            "📖 Wikipedia search",
            value=st.session_state.wiki_enabled,
            help="ON: the lobster can search Wikipedia for facts (two-pass, temp 0.6 for search).",
        )
        st.session_state.wiki_enabled = wiki_enabled

        cooking_enabled = st.toggle(
            "🍳 Cooking pot",
            value=st.session_state.cooking_enabled,
            help="ON: temperature climbs with each answer. The lobster's brain gets progressively cooked.",
        )
        st.session_state.cooking_enabled = cooking_enabled

        # Auto-swap system prompt on mode change, but only if the current
        # prompt is a known default (don't clobber user edits).
        mode_changed = (
            wiki_enabled != st.session_state.prev_wiki
            or cooking_enabled != st.session_state.prev_cooking
        )
        if mode_changed:
            new_default = default_prompt_for_mode(wiki_enabled, cooking_enabled)
            # Check if current prompt matches ANY of the three defaults.
            all_defaults = {WIKI_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT, PLAIN_SYSTEM_PROMPT}
            if st.session_state.system_prompt in all_defaults:
                st.session_state.system_prompt = new_default
                log.info(f"Mode changed -> swapped system prompt (wiki={wiki_enabled}, cooking={cooking_enabled})")
            st.session_state.prev_wiki = wiki_enabled
            st.session_state.prev_cooking = cooking_enabled

        st.divider()

        # --- Cooking status (if cooking is on) ---
        if cooking_enabled:
            st.markdown("### 🍳 Lobster status")
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

        # --- Wiki status (if wiki is on) ---
        if wiki_enabled:
            st.markdown("### 📖 Wikipedia")
            st.caption(f"Search pass runs at fixed temp {WIKI_TEMP}.")
            if st.session_state.last_wiki_result:
                st.caption("Last lookup:")
                with st.expander(st.session_state.last_wiki_result["title"], expanded=False):
                    st.caption(st.session_state.last_wiki_result["extract"][:500] + "...")

        if st.session_state.history:
            if st.button("🧽 Clear memory (forget last turns)", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.divider()
        st.markdown("### 📝 System prompt")
        st.caption("Edit freely. Takes effect on the next question.")

        if st.button("Reset to mode default", use_container_width=True):
            st.session_state.system_prompt = default_prompt_for_mode(wiki_enabled, cooking_enabled)
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
    *2B parameters. Jumping spider territory (if you squint). Remembers the last 4 turns.*
    """)
    # Build mode description from the two toggles.
    mode_tags = []
    if wiki_enabled:
        mode_tags.append("📖 Wikipedia search")
    if cooking_enabled:
        mode_tags.append("🍳 Cooking pot")
    if not mode_tags:
        mode_tags.append("🧠 Plain mode (own knowledge)")
    st.caption(" · ".join(mode_tags))

    if wiki_enabled:
        st.caption(
            "🪸 This lobster feeds on the reef of knowledge known as Wikipedia. "
            "Consider keeping its habitat clean by [donating to the Wikimedia Foundation]"
            "(https://donate.wikimedia.org/w/index.php?title=Special:LandingPage"
            "&country=US&uselang=en&wmf_medium=spontaneous&wmf_source=fr-redir"
            "&wmf_campaign=spontaneous)."
        )
    if cooking_enabled:
        st.caption(
            "Every answer cooks the lobster a little. Let it ramble and watch the temperature climb. "
            "Push it to 5.0° and its brain melts — then it resets and cools off."
        )
    if wiki_enabled and cooking_enabled:
        st.caption(
            "Searches run cool (temp 0.6) for reliability. "
            "The cooked brain reads and answers the Wikipedia results."
        )
    st.caption("Qwen3.5-2B (Q8_0) · Text limit: 500 chars · Text only")

    st.divider()

    # --- Model initialization (cached across reruns) ---
    model_path = download_model_files()
    llm = load_model(model_path)

    # --- Input (inline form, right under the model load message) ---
    # st.chat_input always pins to the bottom of the viewport, which is why
    # the box appeared at the foot of the page. A form with text_input renders
    # inline AND gives us plain Enter-to-send (chat_input needed cmd/ctrl+enter).
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input(
            "Ask the lobster...",
            placeholder="Ask the lobster anything...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🦞 Ask the lobster", use_container_width=True)
        st.caption("Press **Enter** to send · paste is supported")

    if submitted and prompt:
        send_text = prompt.strip()
        truncated = len(send_text) > MAX_TEXT_CHARS
        if truncated:
            send_text = send_text[:MAX_TEXT_CHARS]

        try:
            # Compute cooking temp now so both passes use the same heat level.
            cooking_temp = compute_temperature(st.session_state.heat_seconds)
            total_gen = 0.0  # accumulate generation time for the cooking timer

            if wiki_enabled:
                # --- Wiki two-pass ---
                # Clean history: strip any assistant messages that contain
                # "SEARCH:" so the model never sees prior SEARCH outputs as
                # in-context examples to copy (the main cause of search loops).
                clean_hist = [
                    m for m in st.session_state.history
                    if not (m["role"] == "assistant" and parse_search_command(m["content"]))
                ]

                # Pass 1: decide whether to search. ALWAYS at temp 0.6 for
                # reliable SEARCH output, regardless of cooking mode.
                # Low max_tokens forces a terse SEARCH: line.
                with st.spinner("The lobster is thinking..."):
                    first_pass, gen1 = run_inference(
                        llm, send_text,
                        st.session_state.system_prompt, WIKI_TEMP,
                        clean_hist,
                        max_tokens=32,
                    )
                    total_gen += gen1

                    search_query = parse_search_command(first_pass)
                    log.info(f"Wiki first-pass ({len(first_pass)} chars): search_query={search_query!r}")
                    log.info(f"Wiki first-pass raw: {first_pass[:200]!r}")

                    if search_query:
                        log.info(f"Wiki search requested: '{search_query}'")
                        with st.spinner(f"📖 Looking up '{search_query}' on Wikipedia..."):
                            wiki_result = search_wikipedia(search_query)

                        if wiki_result:
                            st.session_state.last_wiki_result = wiki_result
                            # Pass 2: answer using wiki context. NO history.
                            # If cooking is ON, the answer pass uses the cooking
                            # temp so the "cooked brain" reads Wikipedia.
                            # If cooking is OFF, use 0.6.
                            answer_temp = cooking_temp if cooking_enabled else WIKI_TEMP
                            spinner_msg = f"The lobster is reading Wikipedia... (temp {answer_temp:.2f})" if cooking_enabled else "The lobster is reading Wikipedia..."
                            with st.spinner(spinner_msg):
                                result, gen2 = run_inference(
                                    llm, send_text,
                                    WIKI_ANSWER_PROMPT, answer_temp,
                                    [],  # no history on answer pass
                                    wiki_context=wiki_result["context"],
                                    max_tokens=MAX_TOKENS,
                                )
                                total_gen += gen2
                            # Safety net: if the model STILL emits SEARCH,
                            # show the extract verbatim.
                            if parse_search_command(result):
                                log.warning(f"Wiki answer-pass emitted SEARCH despite WIKI_ANSWER_PROMPT; falling back to extract. raw={result[:120]!r}")
                                result = wiki_result["extract"]
                        else:
                            result = first_pass  # search failed, show whatever the model said
                    else:
                        result = first_pass  # model answered directly, no search

                st.session_state.last_response = result
                st.session_state.last_temp = cooking_temp if cooking_enabled else WIKI_TEMP
                st.session_state.last_truncated = truncated

            else:
                # --- Single-pass (plain or cooking-only) ---
                answer_temp = cooking_temp if cooking_enabled else WIKI_TEMP
                spinner_msg = f"The lobster is thinking... (temp {answer_temp:.2f})" if cooking_enabled else "The lobster is thinking..."
                with st.spinner(spinner_msg):
                    result, total_gen = run_inference(
                        llm, send_text,
                        st.session_state.system_prompt, answer_temp,
                        st.session_state.history,
                    )

                st.session_state.last_response = result
                st.session_state.last_temp = answer_temp
                st.session_state.last_truncated = truncated

            # Advance the cook timer by total generation time (both passes).
            if cooking_enabled:
                st.session_state.heat_seconds += total_gen
                if st.session_state.heat_seconds >= COOK_CYCLE_SECONDS:
                    log.info(
                        f"Lobster fully cooked at {st.session_state.heat_seconds:.1f}s "
                        f"(temp {compute_temperature(st.session_state.heat_seconds):.2f}) — resetting heat"
                    )
                    st.session_state.heat_seconds = 0.0
                    st.toast("🔥 The lobster is fully cooked! Heat reset.", icon="🦞")

            # Common: roll the memory window.
            st.session_state.history.append({"role": "user", "content": send_text})
            st.session_state.history.append({"role": "assistant", "content": result})
            max_msgs = MEMORY_TURNS * 2
            if len(st.session_state.history) > max_msgs:
                st.session_state.history = st.session_state.history[-max_msgs:]

        except Exception as e:
            st.error(f"Inference failed: {e}")
            with st.expander("Traceback"):
                traceback.print_exc(file=sys.stdout)
                st.code(traceback.format_exc())

    # --- Display latest response ---
    if st.session_state.last_response is not None:
        st.markdown("### Response")
        st.markdown(st.session_state.last_response)
        # Show wiki lookup info if wiki was used.
        if wiki_enabled and st.session_state.last_wiki_result:
            w = st.session_state.last_wiki_result
            st.caption(f"📖 Looked up: {w['title']} ({len(w['extract'])} chars from Wikipedia)")
        # Show temperature info if cooking is on, or if temp was not the default.
        if cooking_enabled:
            t_emoji, t_label = doneness_label(st.session_state.last_temp)
            st.caption(f"Generated at temperature {st.session_state.last_temp:.2f} · {t_emoji} {t_label}")
        if st.session_state.get("last_truncated", False):
            st.caption(f"✂️ Your prompt was truncated to {MAX_TEXT_CHARS} chars.")
        turns_held = len(st.session_state.history) // 2
        st.caption(f"🧠 Lobster remembers the last {turns_held} turn(s).")

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
        "Remembers the last 4 turns · toggle Wikipedia/cooking modes in the sidebar."
    )


if __name__ == "__main__":
    main()
