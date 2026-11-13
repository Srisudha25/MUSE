
import streamlit as st
import os
import tempfile
import hashlib

from pathlib import Path
from dotenv import load_dotenv

from rag import process_pdf
from agents import (
    summarizer_agent,
    concept_extractor_agent,
    slide_generator_agent,
    viva_agent
)

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="MUSE | AI Research Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DARK THEME
# ==========================================

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #09090D;
    color: #FFFFFF;
}

[data-testid="stMain"] {
    background-color: #09090D;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 3rem;
    max-width: 1100px;
}

/* TYPOGRAPHY */
h1, h2, h3, h4 {
    color: #FFFFFF !important;
    font-family: 'Segoe UI', sans-serif;
}

p, label, span {
    color: #EAE7F0;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #111018;
    border-right: 1px solid #292238;
}

[data-testid="stSidebar"] * {
    color: #EAE7F0;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(
        135deg,
        #7C3AED,
        #9333EA
    );
    color: #FFFFFF !important;
    border: 1px solid #A78BFA;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #6D28D9;
    border-color: #C4B5FD;
    color: #FFFFFF !important;
}

.stButton > button p {
    color: #FFFFFF !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: #171421;
    border: 1px dashed #8B5CF6;
    border-radius: 12px;
    padding: 12px;
}

/* TEXT INPUT */
.stTextInput input {
    background-color: #171421 !important;
    color: #FFFFFF !important;
    border: 1px solid #45335E;
    border-radius: 8px;
}

.stTextInput input:focus {
    border-color: #A78BFA !important;
}

/* SELECTBOX */
[data-baseweb="select"] > div {
    background-color: #171421 !important;
    color: #FFFFFF !important;
    border: 1px solid #45335E;
}

[data-baseweb="popover"] {
    background-color: #171421;
}

/* EXPANDERS */
[data-testid="stExpander"] {
    background-color: #14111C;
    border: 1px solid #35264E;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #FFFFFF;
}

/* DOWNLOAD BUTTONS */
.stDownloadButton > button {
    background-color: #211638;
    color: #FFFFFF !important;
    border: 1px solid #8B5CF6;
    border-radius: 8px;
}

.stDownloadButton > button:hover {
    background-color: #352050;
    border-color: #A78BFA;
}

/* ALERTS */
[data-testid="stAlert"] {
    background-color: #171421;
    color: #FFFFFF;
    border: 1px solid #45335E;
    border-radius: 10px;
}

/* DIVIDERS */
hr {
    border-color: #302440;
}

/* CAPTIONS */
[data-testid="stCaptionContainer"] p {
    color: #A7A0B5 !important;
}

/* CUSTOM HEADER */
.muse-title {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: #FFFFFF;
    margin-bottom: 0;
}

.muse-subtitle {
    font-size: 0.85rem;
    letter-spacing: 0.18em;
    color: #A78BFA;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

.muse-description {
    font-size: 1.1rem;
    color: #BDB6CA;
    margin-bottom: 2rem;
}

/* FEATURE CARDS */
.feature-card {
    background-color: #15121E;
    border: 1px solid #302440;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 15px;
    min-height: 130px;
}

.feature-card h3 {
    color: #C4B5FD !important;
    font-size: 1.1rem;
    margin-bottom: 8px;
}

.feature-card p {
    color: #BDB6CA;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================

if "paper_ready" not in st.session_state:
    st.session_state.paper_ready = False

if "paper_name" not in st.session_state:
    st.session_state.paper_name = None

if "paper_hash" not in st.session_state:
    st.session_state.paper_hash = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="muse-title">MUSE<span style="color:#A78BFA;">.</span></div>
<div class="muse-subtitle">
MULTIMODAL UNDERSTANDING & SCHOLARLY EXPLORATION
</div>
<div class="muse-description">
Your intelligent research companion.
Transform complex papers into clear insights,
presentation content and confident viva preparation.
</div>
""", unsafe_allow_html=True)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## ✦ MUSE")
    st.caption("YOUR RESEARCH WORKSPACE")

    st.divider()

    st.markdown("### Research Paper")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.caption(f"Selected: {uploaded_file.name}")

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if st.session_state.paper_hash != file_hash:
            st.session_state.paper_ready = False

        if st.button(
            "✦ Process Paper",
            use_container_width=True
        ):

            with st.spinner("Reading and indexing your paper..."):

                temp_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(file_bytes)
                        temp_path = temp_file.name

                    process_pdf(temp_path)

                    st.session_state.paper_ready = True
                    st.session_state.paper_name = uploaded_file.name
                    st.session_state.paper_hash = file_hash
                    st.session_state.chat_history = []

                    st.success("Paper processed successfully!")

                except Exception as e:
                    st.error(f"Processing failed: {e}")

                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

    if st.session_state.paper_ready:

        st.divider()

        st.markdown("### Active Document")

        st.success(st.session_state.paper_name)

        st.caption("Ready for AI analysis")

    st.divider()

    st.caption("MUSE © 2026 | Research Intelligence")


# ==========================================
# MAIN INTERFACE
# ==========================================

if not st.session_state.paper_ready:

    st.markdown("## Explore what MUSE can do")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="feature-card">
            <h3>✦ Research Summaries</h3>
            <p>Turn complex research papers into
            clear, structured academic summaries.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>✦ Presentation Builder</h3>
            <p>Transform research findings into
            organised, presentation-ready slides.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="feature-card">
            <h3>✦ Concept Extraction</h3>
            <p>Identify methodologies, algorithms,
            contributions and important findings.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>✦ Viva Preparation</h3>
            <p>Prepare for examinations with
            AI-generated questions and model answers.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Upload and process a research paper "
        "from the sidebar to begin."
    )

else:

    st.markdown("## Your Research Studio")

    st.caption(
        f"Currently analysing: {st.session_state.paper_name}"
    )

    st.markdown("### Choose your task")

    task = st.selectbox(
        "What would you like MUSE to do?",
        [
            "Summarise Research Paper",
            "Extract Key Concepts",
            "Generate Presentation Slides",
            "Prepare for Viva"
        ]
    )

    custom_query = st.text_input(
        "Additional instructions (optional)",
        placeholder="e.g. Explain the methodology in detail..."
    )

    st.write("")

    if st.button(
        "✦ Generate Results",
        use_container_width=True
    ):

        query = custom_query.strip()

        if task == "Summarise Research Paper":

            if not query:
                query = "Summarise this research paper"

            agent = summarizer_agent

        elif task == "Extract Key Concepts":

            if not query:
                query = "Extract the key concepts"

            agent = concept_extractor_agent

        elif task == "Generate Presentation Slides":

            if not query:
                query = "Generate presentation slides"

            agent = slide_generator_agent

        else:

            if not query:
                query = "Generate likely viva questions"

            agent = viva_agent

        with st.spinner("MUSE is analysing your research..."):

            try:
                result = agent(query)

                st.session_state.chat_history.append({
                    "task": task,
                    "query": query,
                    "result": result
                })

                st.success("Analysis completed!")

            except Exception as e:
                st.error(f"Generation failed: {e}")

    # ======================================
    # RESULTS
    # ======================================

    if st.session_state.chat_history:

        st.divider()

        st.markdown("## Your Results")

        for i, item in enumerate(
            reversed(st.session_state.chat_history)
        ):

            with st.expander(
                f"✦ {item['task']}",
                expanded=(i == 0)
            ):

                st.caption(
                    f"Request: {item['query']}"
                )

                st.markdown(item["result"])

                st.download_button(
                    label="↓ Download Result",
                    data=item["result"],
                    file_name=f"muse_result_{i+1}.md",
                    mime="text/markdown",
                    key=f"download_{i}"
                )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "MUSE — Multimodal Understanding & Scholarly Exploration"
)
