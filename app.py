"""
app.py
------
GENE — Generative ENgine Agent

Streamlit UI for the GENE AI Writing Assistant.

Architecture:
    User
      ↓
    Streamlit UI
      ↓
    GeneAgent
      ↓
    Hugging Face Transformers
      ↓
    Qwen2.5-Instruct
      ↓
    Generated Answer

Important:
- agent.py remains the intelligence/backend layer.
- assets/style.css remains the main UI stylesheet.
- This file only controls the Streamlit interface and routing.
"""

from pathlib import Path
from textwrap import dedent
import base64
import html
import time

import streamlit as st

from agent import (
    GeneAgent,
    GenerationParams,
    TASK_INSTRUCTIONS,
    DEFAULT_MODEL_NAME,
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="GENE — Your AI Writing Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# THEME VARIABLES
# ============================================================================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"


DARK_VARS = """
:root {
    --bg: #060a14;
    --bg-2: #0a0f1e;
    --panel: #0d1526;
    --panel-2: #0f1830;

    --border: #1b2740;
    --border-soft: #16203a;

    --text: #eaf0ff;
    --text-dim: #8a95b3;
    --text-faint: #5b6684;

    --blue: #5b8cff;
    --purple: #9b6bff;
    --green: #3ddc97;
    --orange: #f5a623;
    --red: #ff5c7a;
}
"""


LIGHT_VARS = """
:root {
    --bg: #f4f6fb;
    --bg-2: #ffffff;
    --panel: #ffffff;
    --panel-2: #f0f3fa;

    --border: #e1e6f0;
    --border-soft: #eaeef6;

    --text: #161c2d;
    --text-dim: #5c6785;
    --text-faint: #9aa3bd;

    --blue: #3a6bef;
    --purple: #8355e8;
    --green: #1fae7d;
    --orange: #d4880f;
    --red: #e6415f;
}
"""


# ============================================================================
# LOGO
# ============================================================================

def get_logo_html(size: int = 48) -> str:
    """
    Load the AX/GENE logo from assets.

    Supported files:
        assets/logo.png
        assets/logo.jpg
        assets/logo.jpeg
        assets/logo.svg
        assets/logo.webp

    If no image exists, AX text is used as a fallback.
    """

    assets_dir = Path(__file__).parent / "assets"

    for name in (
        "logo.png",
        "logo.jpg",
        "logo.jpeg",
        "logo.svg",
        "logo.webp",
    ):
        path = assets_dir / name

        if not path.exists():
            continue

        ext = path.suffix.lower().replace(".", "")

        if ext == "svg":
            mime = "svg+xml"
        elif ext in {"jpg", "jpeg"}:
            mime = "jpeg"
        else:
            mime = ext

        try:
            encoded = base64.b64encode(
                path.read_bytes()
            ).decode("utf-8")

            return (
                f'<img '
                f'src="data:image/{mime};base64,{encoded}" '
                f'width="{size}" '
                f'height="{size}" '
                f'style="'
                f'width:{size}px;'
                f'height:{size}px;'
                f'object-fit:contain;'
                f'border-radius:8px;'
                f'" />'
            )

        except Exception:
            continue

    return (
        f'<span '
        f'class="gene-brand-mark" '
        f'style="font-size:{int(size * 0.55)}px;">'
        f'AX'
        f'</span>'
    )


# ============================================================================
# HTML RENDER HELPER
# ============================================================================

def render_html(content: str) -> None:
    """Render custom HTML directly using Streamlit's HTML renderer."""
    st.html(dedent(content).strip())
# ============================================================================
# CSS LOADER
# ============================================================================

def load_css() -> None:
    """
    Load theme variables and assets/style.css.
    """

    variables = (
        DARK_VARS
        if st.session_state.theme == "dark"
        else LIGHT_VARS
    )

    render_html(
        f"""
        <style>
        {variables}
        </style>
        """
    )

    css_path = (
        Path(__file__).parent
        / "assets"
        / "style.css"
    )

    if css_path.exists():
        try:
            css = css_path.read_text(
                encoding="utf-8"
            )

            render_html(
                f"""
                <style>
                {css}
                </style>
                """
            )

        except Exception as exc:
            st.warning(
                f"Unable to load style.css: {exc}"
            )


load_css()


# ============================================================================
# EXTRA STREAMLIT UI CSS
# ============================================================================

render_html(
    """
    <style>

    /* Prevent horizontal overflow */
    .main .block-container {
        overflow-x: hidden;
    }

    /* Better spacing */
    [data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }

    /* Safe HTML wrapper */
    .gene-safe-html {
        width: 100%;
    }

    /* Message cards */
    .gene-message-card {
        width: 100%;
        margin-bottom: 18px;
    }

    /* User message */
    .gene-user-message {
        background: linear-gradient(
            135deg,
            rgba(91, 140, 255, 0.16),
            rgba(91, 140, 255, 0.06)
        );
        border: 1px solid rgba(91, 140, 255, 0.24);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    /* AI message */
    .gene-ai-message {
        background: var(--panel-2);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }

    /* Message header */
    .gene-message-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 9px;
    }

    .gene-message-name {
        font-size: 12px;
        font-weight: 700;
        color: var(--blue);
    }

    .gene-ai-name {
        color: var(--purple);
    }

    /* Message text */
    .gene-message-content {
        color: var(--text);
        font-size: 14px;
        line-height: 1.7;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: anywhere;
    }

    /* Task badge */
    .gene-task-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 20px;
        background: rgba(91, 140, 255, 0.12);
        color: var(--blue);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    /* Empty state */
    .gene-empty-container {
        text-align: center;
        padding: 58px 20px;
    }

    .gene-empty-title {
        color: var(--text);
        font-size: 18px;
        font-weight: 700;
        margin-top: 14px;
    }

    .gene-empty-subtitle {
        color: var(--text-dim);
        font-size: 13px;
        margin-top: 6px;
    }

    /* Code blocks */
    [data-testid="stCodeBlock"] {
        border-radius: 10px;
    }

    /* AI Markdown output */
    .gene-output-markdown {
        color: var(--text);
        line-height: 1.7;
        font-size: 14px;
        margin-bottom: 12px;
    }

    .gene-output-markdown h1,
    .gene-output-markdown h2,
    .gene-output-markdown h3 {
        color: var(--text);
    }

    .gene-output-markdown code {
        color: var(--blue);
    }

    /* Composer */
    .gene-composer-label {
        color: var(--text-dim);
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    /* Disclaimer */
    .gene-disclaimer {
        text-align: center;
        color: var(--text-faint);
        font-size: 11px;
        margin-top: 12px;
    }

    /* Mobile */
    @media (max-width: 800px) {
        .gene-stats-row {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }

    </style>
    """
)


# ============================================================================
# CACHED AGENT
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_agent(model_name: str) -> GeneAgent:
    """
    Create and cache one GeneAgent per selected model.

    Actual model loading and generation remain inside agent.py.
    """

    return GeneAgent(
        model_name=model_name
    )


# ============================================================================
# SESSION STATE
# ============================================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "saved" not in st.session_state:
    st.session_state.saved = []

if "view" not in st.session_state:
    st.session_state.view = "chat"


TASKS = list(
    TASK_INSTRUCTIONS.keys()
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_text(value) -> str:
    """
    Convert arbitrary values into safe display text.
    """

    if value is None:
        return ""

    return str(value)


def render_plain_message(text: str) -> None:
    """
    Render generated AI text using Streamlit Markdown.

    The AI output is intentionally NOT inserted into HTML.
    This allows Markdown, code blocks, lists, headings, etc.
    to render correctly.
    """

    text = safe_text(text)

    if not text.strip():

        render_html(
            """
            <span
                style="color:var(--text-faint);"
            >
                No output generated.
            </span>
            """
        )

        return

    st.markdown(text)


def build_generation_params() -> GenerationParams:
    """
    Build GenerationParams from current sidebar settings.
    """

    return GenerationParams(
        max_new_tokens=int(
            st.session_state.get(
                "max_tokens",
                256,
            )
        ),
        temperature=float(
            st.session_state.get(
                "temperature",
                0.7,
            )
        ),
        top_p=float(
            st.session_state.get(
                "top_p",
                0.9,
            )
        ),
        top_k=int(
            st.session_state.get(
                "top_k",
                50,
            )
        ),
        num_return_sequences=int(
            st.session_state.get(
                "num_outputs",
                1,
            )
        ),
        repetition_penalty=float(
            st.session_state.get(
                "repetition_penalty",
                1.1,
            )
        ),
        do_sample=bool(
            st.session_state.get(
                "do_sample",
                True,
            )
        ),
    )


def generate_response(
    prompt: str,
    task: str,
    model_name: str,
):
    """
    Central generation function.

    All actual model intelligence stays inside agent.py.
    """

    agent = get_agent(
        model_name
    )

    params = build_generation_params()

    # Load model only when required.
    if not agent.is_loaded:

        with st.spinner(
            f"Loading {model_name}..."
        ):
            agent.load()

    start_time = time.perf_counter()

    with st.spinner(
        "GENE is thinking..."
    ):

        response = agent.generate(
            prompt=prompt,
            task=task,
            params=params,
        )

    measured_time = (
        time.perf_counter()
        - start_time
    )

    return response, measured_time


# ============================================================================
# MESSAGE RENDERER
# ============================================================================

def render_message(
    item: dict,
    idx: int,
    saved: bool = False,
) -> None:
    """
    Render one prompt/response pair.

    IMPORTANT:
    - User text is HTML escaped.
    - AI text is rendered separately as Markdown.
    - AI text is NEVER placed directly inside HTML.
    """

    prompt_text = safe_text(
        item.get(
            "prompt",
            "",
        )
    )

    output_text = safe_text(
        item.get(
            "output",
            "",
        )
    )

    task_text = safe_text(
        item.get(
            "task",
            "General",
        )
    )

    model_text = safe_text(
        item.get(
            "model",
            DEFAULT_MODEL_NAME,
        )
    )

    # ------------------------------------------------------------------
    # USER MESSAGE
    # ------------------------------------------------------------------

    escaped_prompt = html.escape(
        prompt_text
    )

    render_html(
        f"""
        <div class="gene-msg-row">

            <div class="gene-avatar user">
                🧑
            </div>

            <div class="gene-bubble user">

                <div class="gene-bubble-head">
                    <span class="gene-bubble-name">
                        You
                    </span>
                </div>

                <div class="gene-bubble-text">
                    {escaped_prompt}
                </div>

            </div>

        </div>
        """
    )

    # ------------------------------------------------------------------
    # GENE HEADER
    # ------------------------------------------------------------------

    escaped_task = html.escape(
        task_text
    )

    render_html(
        f"""
        <div class="gene-msg-row">

            <div class="gene-avatar bot">
                AX
            </div>

            <div style="flex:1;">

                <div class="gene-bubble bot">

                    <div class="gene-bubble-head">

                        <span class="gene-bubble-name">
                            GENE
                        </span>

                        <span class="gene-badge">
                            {escaped_task}
                        </span>

                    </div>

                </div>

            </div>

        </div>
        """
    )

    # ------------------------------------------------------------------
    # AI OUTPUT
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="gene-output-markdown">
        """
    )

    render_plain_message(
        output_text
    )

    render_html(
        """
        </div>
        """
    )

    # ------------------------------------------------------------------
    # ACTION BUTTONS
    # ------------------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    # ------------------------------------------------------------------
    # REGENERATE
    # ------------------------------------------------------------------

    with c1:

        if not saved:

            if st.button(
                "🔁 Regenerate",
                key=f"regen_{idx}",
                use_container_width=True,
            ):

                try:

                    response, measured_time = (
                        generate_response(
                            prompt=prompt_text,
                            task=task_text,
                            model_name=model_text,
                        )
                    )

                    if response.success:

                        response_time = getattr(
                            response,
                            "elapsed_seconds",
                            measured_time,
                        )

                        if (
                            0
                            <= idx
                            < len(
                                st.session_state.history
                            )
                        ):

                            st.session_state.history[
                                idx
                            ] = {
                                "task": response.task,
                                "prompt": prompt_text,
                                "output": response.output,
                                "elapsed": response_time,
                                "model": model_text,
                            }

                        st.rerun()

                    else:

                        st.error(
                            safe_text(
                                response.error
                            )
                        )

                except Exception as exc:

                    st.error(
                        f"Generation error: {exc}"
                    )

    # ------------------------------------------------------------------
    # DOWNLOAD
    # ------------------------------------------------------------------

    with c2:

        st.download_button(
            "⬇️ Download",
            data=output_text,
            file_name="gene_output.txt",
            mime="text/plain",
            key=f"download_{idx}_{saved}",
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    with c3:

        if not saved:

            if st.button(
                "❤️ Save",
                key=f"save_{idx}",
                use_container_width=True,
            ):

                st.session_state.saved.append(
                    dict(item)
                )

                st.toast(
                    "Saved to Saved Prompts"
                )

    # ------------------------------------------------------------------
    # REMOVE
    # ------------------------------------------------------------------

    with c4:

        if saved:

            if st.button(
                "🗑️ Remove",
                key=f"remove_{idx}",
                use_container_width=True,
            ):

                if (
                    0
                    <= idx
                    < len(
                        st.session_state.saved
                    )
                ):

                    st.session_state.saved.pop(
                        idx
                    )

                st.rerun()

    st.divider()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    # ------------------------------------------------------------------
    # BRANDING
    # ------------------------------------------------------------------

    render_html(
        f"""
        <div class="gene-brand-card">
            {get_logo_html(64)}
        </div>
        """
    )

    render_html(
        """
        <p class="gene-title">
            GENE
        </p>
        """
    )

    render_html(
        """
        <p class="gene-subtitle">
            Generative ENgine Agent
        </p>
        """
    )

    render_html(
        """
        <div class="gene-settings-heading">
            GENERATION SETTINGS
        </div>
        """
    )

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------

    model_name = st.selectbox(
        "Model",
        options=[
            DEFAULT_MODEL_NAME,
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
        ],
        index=0,
        key="model_selector",
    )

    # ------------------------------------------------------------------
    # MAX TOKENS
    # ------------------------------------------------------------------

    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=32,
        max_value=1024,
        value=256,
        step=32,
        key="max_tokens",
    )

    # ------------------------------------------------------------------
    # TEMPERATURE
    # ------------------------------------------------------------------

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.05,
        key="temperature",
    )

    # ------------------------------------------------------------------
    # TOP P
    # ------------------------------------------------------------------

    top_p = st.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        key="top_p",
    )

    # ------------------------------------------------------------------
    # TOP K
    # ------------------------------------------------------------------

    top_k = st.slider(
        "Top-k",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        key="top_k",
    )

    # ------------------------------------------------------------------
    # REPETITION PENALTY
    # ------------------------------------------------------------------

    repetition_penalty = st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.1,
        step=0.05,
        key="repetition_penalty",
    )

    # ------------------------------------------------------------------
    # SAMPLING
    # ------------------------------------------------------------------

    do_sample = st.checkbox(
        "Sampling (do_sample)",
        value=True,
        key="do_sample",
    )

    # ------------------------------------------------------------------
    # OUTPUT COUNT
    # ------------------------------------------------------------------

    num_return_sequences = st.number_input(
        "Number of Outputs",
        min_value=1,
        max_value=3,
        value=1,
        step=1,
        key="num_outputs",
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # NEW CHAT
    # ------------------------------------------------------------------

    if st.button(
        "🆕  New Chat",
        key="nav_new",
        use_container_width=True,
    ):

        st.session_state.history = []
        st.session_state.view = "chat"

        st.rerun()

    # ------------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------------

    if st.button(
        "🕒  History",
        key="nav_history",
        use_container_width=True,
    ):

        st.session_state.view = "history"

        st.rerun()

    # ------------------------------------------------------------------
    # SAVED
    # ------------------------------------------------------------------

    if st.button(
        "⭐  Saved Prompts",
        key="nav_saved",
        use_container_width=True,
    ):

        st.session_state.view = "saved"

        st.rerun()

    # ------------------------------------------------------------------
    # SETTINGS
    # ------------------------------------------------------------------

    if st.button(
        "⚙️  Settings",
        key="nav_settings",
        use_container_width=True,
    ):

        st.session_state.view = "settings"

        st.rerun()

    # ------------------------------------------------------------------
    # ABOUT
    # ------------------------------------------------------------------

    if st.button(
        "ℹ️  About GENE",
        key="nav_about",
        use_container_width=True,
    ):

        st.session_state.view = "about"

        st.rerun()

    # ------------------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="sidebar-footer">
            © 2026 GENE — All rights reserved.
        </div>
        """
    )


# ============================================================================
# TOP BAR
# ============================================================================

top_left, top_right = st.columns(
    [5, 1],
    vertical_alignment="center",
)


# --------------------------------------------------------------------------
# TOP LEFT
# --------------------------------------------------------------------------

with top_left:

    render_html(
        f"""
        <div class="gene-topbar">

            <div class="gene-topbar-brand">

                <div class="gene-topbar-mark">
                    {get_logo_html(30)}
                </div>

                <div>

                    <p class="gene-topbar-title">
                        GENE
                    </p>

                    <p class="gene-topbar-subtitle">
                        Your AI Writing Assistant
                    </p>

                </div>

            </div>

        </div>
        """
    )


# --------------------------------------------------------------------------
# TOP RIGHT
# --------------------------------------------------------------------------

with top_right:

    if st.session_state.theme == "dark":
        toggle_label = "☀️ Light Mode"
    else:
        toggle_label = "🌙 Dark Mode"

    if st.button(
        toggle_label,
        key="theme_toggle",
        use_container_width=True,
    ):

        if st.session_state.theme == "dark":
            st.session_state.theme = "light"
        else:
            st.session_state.theme = "dark"

        st.rerun()


# ============================================================================
# STATISTICS
# ============================================================================

latest = (
    st.session_state.history[0]
    if st.session_state.history
    else None
)


if latest:

    latest_output = safe_text(
        latest.get(
            "output",
            "",
        )
    )

    words = len(
        latest_output.split()
    )

    chars = len(
        latest_output
    )

    tokens_est = round(
        words * 1.35
    )

    runtime_value = latest.get(
        "elapsed",
        0.0,
    )

    try:

        runtime = (
            f"{float(runtime_value):.2f}s"
        )

    except Exception:

        runtime = "—"

    latest_model = safe_text(
        latest.get(
            "model",
            model_name,
        )
    )

    model_short = (
        latest_model
        .split("/")[-1]
        .replace(
            "-Instruct",
            "",
        )
    )

    task_label = safe_text(
        latest.get(
            "task",
            "—",
        )
    )

else:

    words = 0
    chars = 0
    tokens_est = 0
    runtime = "0.00s"

    model_short = (
        model_name
        .split("/")[-1]
        .replace(
            "-Instruct",
            "",
        )
    )

    task_label = "—"


# Escape dynamic values before placing them in HTML.

safe_model_short = html.escape(
    model_short
)

safe_task_label = html.escape(
    task_label
)


render_html(
    f"""
    <div class="gene-stats-row">

        <div
            class="gene-stat-card"
            style="--accent:var(--green)"
        >

            <div class="gene-stat-head">
                <span class="gene-stat-dot"></span>
                WORDS
            </div>

            <div class="gene-stat-value">
                {words}
            </div>

        </div>


        <div
            class="gene-stat-card"
            style="--accent:var(--purple)"
        >

            <div class="gene-stat-head">
                <span class="gene-stat-dot"></span>
                CHARACTERS
            </div>

            <div class="gene-stat-value">
                {chars}
            </div>

        </div>


        <div
            class="gene-stat-card"
            style="--accent:var(--blue)"
        >

            <div class="gene-stat-head">
                <span class="gene-stat-dot"></span>
                TOKENS (EST.)
            </div>

            <div class="gene-stat-value">
                {tokens_est}
            </div>

        </div>


        <div
            class="gene-stat-card"
            style="--accent:var(--orange)"
        >

            <div class="gene-stat-head">
                <span class="gene-stat-dot"></span>
                RUNTIME
            </div>

            <div class="gene-stat-value">
                {html.escape(runtime)}
            </div>

        </div>


        <div
            class="gene-stat-card"
            style="--accent:var(--blue)"
        >

            <div class="gene-stat-head">
                <span class="gene-stat-dot"></span>
                MODEL
            </div>

            <div class="gene-stat-value gene-stat-value-sm">
                {safe_model_short}
            </div>

        </div>


        <div
            class="gene-stat-card"
            style="--accent:var(--blue)"
        >

            <div class="gene-stat-head">
                <span class="gene-stat-dot"></span>
                TASK
            </div>

            <div class="gene-stat-value gene-stat-value-sm">
                {safe_task_label}
            </div>

        </div>

    </div>
    """
)


# ============================================================================
# CURRENT VIEW
# ============================================================================

view = st.session_state.view


# ============================================================================
# CHAT VIEW
# ============================================================================

if view == "chat":

    # ------------------------------------------------------------------
    # EMPTY CHAT
    # ------------------------------------------------------------------

    if not st.session_state.history:

        render_html(
            f"""
            <div class="gene-chat-panel">

                <div class="gene-empty">

                    <div class="mark">
                        {get_logo_html(48)}
                    </div>

                    <p>
                        Ask GENE to write a story, email,
                        summary, explanation, or creative text.
                    </p>

                </div>

            </div>
            """
        )

    # ------------------------------------------------------------------
    # CHAT HISTORY
    # ------------------------------------------------------------------

    else:

        for i, item in enumerate(
            st.session_state.history
        ):

            render_message(
                item=item,
                idx=i,
            )

    # ------------------------------------------------------------------
    # COMPOSER LABEL
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="gene-composer-label">
            Prompt
        </div>
        """
    )

    # ------------------------------------------------------------------
    # COMPOSER
    # ------------------------------------------------------------------

    composer_left, composer_right = st.columns(
        [4, 1],
        vertical_alignment="bottom",
    )

    with composer_left:

        prompt = st.text_area(
            "Prompt",
            placeholder=(
                "Type your prompt here..."
            ),
            height=110,
            label_visibility="collapsed",
            key="prompt_input",
        )

    with composer_right:

        task = st.selectbox(
            "Task",
            TASKS,
            label_visibility="collapsed",
            key="task_select",
        )

        send_clicked = st.button(
            "🚀 Send",
            use_container_width=True,
            key="send_btn",
            type="primary",
        )

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------

    if send_clicked:

        clean_prompt = safe_text(
            prompt
        ).strip()

        if not clean_prompt:

            st.warning(
                "Please enter a prompt before sending."
            )

        else:

            try:

                response, measured_time = (
                    generate_response(
                        prompt=clean_prompt,
                        task=task,
                        model_name=model_name,
                    )
                )

                if response.success:

                    response_time = getattr(
                        response,
                        "elapsed_seconds",
                        measured_time,
                    )

                    st.session_state.history.insert(
                        0,
                        {
                            "task": response.task,
                            "prompt": clean_prompt,
                            "output": response.output,
                            "elapsed": response_time,
                            "model": model_name,
                        },
                    )

                    st.rerun()

                else:

                    st.error(
                        safe_text(
                            response.error
                        )
                    )

            except Exception as exc:

                st.error(
                    "GENE could not generate a response: "
                    f"{exc}"
                )

    # ------------------------------------------------------------------
    # DISCLAIMER
    # ------------------------------------------------------------------

    render_html(
        """
        <p class="gene-disclaimer">
            GENE can make mistakes.
            Please verify important information.
        </p>
        """
    )


# ============================================================================
# HISTORY VIEW
# ============================================================================

elif view == "history":

    st.markdown(
        "## History"
    )

    if not st.session_state.history:

        st.info(
            "No generations yet. "
            "Go to New Chat to get started."
        )

    else:

        for i, item in enumerate(
            st.session_state.history
        ):

            render_message(
                item=item,
                idx=i,
            )


# ============================================================================
# SAVED PROMPTS VIEW
# ============================================================================

elif view == "saved":

    st.markdown(
        "## Saved Prompts"
    )

    if not st.session_state.saved:

        st.info(
            "Nothing saved yet. "
            "Use ❤️ Save on a response to keep it here."
        )

    else:

        for i, item in enumerate(
            st.session_state.saved
        ):

            render_message(
                item=item,
                idx=i,
                saved=True,
            )


# ============================================================================
# SETTINGS VIEW
# ============================================================================

elif view == "settings":

    st.markdown(
        "## Settings"
    )

    st.markdown(
        """
        Configure GENE's generation behavior from the
        sidebar. Changes apply to the next generation.
        """
    )

    st.markdown(
        "### Current Configuration"
    )

    settings_table = {
        "Model": model_name,
        "Max New Tokens": max_new_tokens,
        "Temperature": temperature,
        "Top-p": top_p,
        "Top-k": top_k,
        "Repetition Penalty": repetition_penalty,
        "Sampling": (
            "On"
            if do_sample
            else "Off"
        ),
        "Outputs": int(
            num_return_sequences
        ),
    }

    for key, value in settings_table.items():

        col1, col2 = st.columns(
            [2, 4]
        )

        with col1:

            st.markdown(
                f"**{key}**"
            )

        with col2:

            st.write(value)


# ============================================================================
# ABOUT VIEW
# ============================================================================

elif view == "about":

    st.markdown(
        "## About GENE"
    )

    st.markdown(
        """
        **GENE — Generative ENgine Agent**

        GENE is an AI writing assistant built around a
        task-aware AI agent and a Hugging Face
        Transformers text-generation pipeline.
        """
    )

    st.markdown(
        "### Architecture"
    )

    st.code(
        """
User
  ↓
Streamlit UI
  ↓
GeneAgent
  ↓
Hugging Face Transformers
  ↓
Qwen2.5-Instruct
  ↓
Generated Answer
  ↓
GENE UI
        """.strip(),
        language="text",
    )

    st.markdown(
        "### Supported Tasks"
    )

    for task_name in TASKS:

        st.markdown(
            f"- **{task_name}**"
        )

    st.markdown(
        f"""
        ### Default Model

        `{DEFAULT_MODEL_NAME}`

        The model can be changed from the sidebar.
        """
    )

    st.info(
        "GENE is designed to provide useful AI-generated "
        "responses, but important information should always "
        "be verified."
    )