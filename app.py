"""
ResearchMate AI – Intelligent Research Paper Summarizer with Cloud AI Assistant
==============================================================================
A clean, professional academic web application that:
1. Extracts text from PDF research papers and produces an 8-part structured summary.
2. Features a ☁️ Cloud AI Assistant that opens on demand to answer questions about
   the uploaded paper with complete knowledge across all academic and general topics.

Built with Streamlit & Google Gemini API (official google-genai SDK).
"""

# ── Standard library imports ──────────────────────────────────────────
import os
import textwrap
import time
from pathlib import Path

# ── Third-party imports ───────────────────────────────────────────────
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

# ── Load environment variables (.env) ─────────────────────────────────
env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_file, override=True)
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 25                       # Maximum allowed PDF size in MB
MAX_CHARS_FOR_API = 120_000                 # Character limit sent to Gemini (~30k tokens)
CANDIDATE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]


# =====================================================================
#  PAGE CONFIGURATION & PREMIUM CSS
# =====================================================================

st.set_page_config(
    page_title="ResearchMate AI – Research Paper Summarizer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 950px;
}

/* ── Hero Header ───────────────────────────────────────── */
.hero-wrapper {
    text-align: center;
    padding: 1.8rem 1rem 1.2rem 1rem;
    background: linear-gradient(180deg, #f8faff 0%, #ffffff 100%);
    border-radius: 16px;
    border: 1px solid #e8eaf6;
    margin-bottom: 1.2rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #ede7f6;
    color: #5e35b1;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 0.6rem;
    letter-spacing: 0.3px;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0d47a1 0%, #512da8 50%, #7c4dff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -0.8px;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #455a64;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.hero-tagline {
    font-size: 0.92rem;
    color: #78909c;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ── Floating / Header Cloud Bar ────────────────────────── */
.cloud-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(135deg, #ede7f6 0%, #e1f5fe 100%);
    border: 1px solid #c5cae9;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.2rem;
}
.cloud-bar-left {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.cloud-icon {
    font-size: 1.8rem;
    animation: floatCloud 3s ease-in-out infinite;
}
@keyframes floatCloud {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-3px); }
    100% { transform: translateY(0px); }
}

/* ── Cloud Chat Container ───────────────────────────────── */
.cloud-chat-card {
    background: #ffffff;
    border: 2px solid #7c4dff;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(124, 77, 255, 0.12);
}
.cloud-chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #ede7f6;
    padding-bottom: 0.8rem;
    margin-bottom: 1rem;
}
.cloud-chat-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a237e;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Step pills ────────────────────────────────────────── */
.steps-row {
    display: flex;
    justify-content: center;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.step-pill {
    background: #ffffff;
    border: 1px solid #e0e7ff;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    min-width: 140px;
    flex: 1;
    max-width: 180px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.step-pill .step-num {
    display: inline-block;
    background: #5e35b1;
    color: #fff;
    width: 24px;
    height: 24px;
    line-height: 24px;
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.75rem;
    margin-bottom: 0.3rem;
}
.step-pill .step-label {
    font-size: 0.82rem;
    color: #37474f;
    font-weight: 600;
}

/* ── File Badge ────────────────────────────────────────── */
.file-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: #e8eaf6;
    color: #1a237e;
    border: 1px solid #c5cae9;
    border-radius: 10px;
    padding: 0.6rem 1.1rem;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

/* ── Buttons ───────────────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #1a237e 0%, #512da8 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(26,35,126,0.2);
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(81,45,168,0.35);
    color: white;
}

div.stDownloadButton > button {
    background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(13,71,161,0.2);
}

.cloud-btn-trigger > button {
    background: linear-gradient(135deg, #673ab7 0%, #7c4dff 100%) !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}

.app-footer {
    text-align: center;
    color: #90a4ae;
    font-size: 0.82rem;
    padding: 2.5rem 0 1rem 0;
    border-top: 1px solid #eceff1;
    margin-top: 2.5rem;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
#  SESSION STATE INITIALIZATION
# =====================================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analyzed_file_name" not in st.session_state:
    st.session_state.analyzed_file_name = None

if "analyzed_file_size" not in st.session_state:
    st.session_state.analyzed_file_size = None

if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""

# Track if Cloud Chatbot popup/panel is active
if "show_cloud_chat" not in st.session_state:
    st.session_state.show_cloud_chat = False

if "cloud_chat_messages" not in st.session_state:
    st.session_state.cloud_chat_messages = [
        {
            "role": "assistant",
            "content": (
                "☁️ **Hello! I am your Cloud Research Assistant.**\n\n"
                "I have knowledge of **everything** — from any field of science, math, AI, and code, "
                "to history and philosophy.\n\n"
                "📄 **Upload a PDF** above, and I will instantly answer questions directly about your paper, "
                "or you can ask me anything about any other topic right now!"
            )
        }
    ]


# =====================================================================
#  ROBUST GEMINI CLIENT & QUERY ENGINE
# =====================================================================

def get_gemini_client():
    """Returns an authenticated Gemini client or None with an error."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "your_api_key_here":
        st.error(
            "⚠️ **Gemini API key not configured.**\n\n"
            "Please add your key to `.env`:\n\n"
            "```\nGEMINI_API_KEY=your_key_here\n```\n\n"
            "Get a key at [Google AI Studio](https://aistudio.google.com/apikey)."
        )
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini Client: {e}")
        return None


def query_gemini_robust(client, prompt, system_instruction=None):
    """
    Query Gemini with candidate fallback.
    Tries gemini-3.1-flash-lite first for instant response,
    then gemini-3.8-flash, then gemini-3.5-flash-lite.
    """
    last_error = None
    combined_prompt = prompt
    if system_instruction:
        combined_prompt = f"System Instruction:\n{system_instruction}\n\nTask:\n{prompt}"

    for model_name in CANDIDATE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=combined_prompt,
            )
            if response and response.text:
                return response.text, model_name, None
        except Exception as e:
            last_error = e
            continue

    return None, None, last_error


def extract_text_from_pdf(uploaded_file):
    """Extract readable text from uploaded PDF using pypdf."""
    try:
        reader = PdfReader(uploaded_file)
    except Exception:
        return "", 0, "❌ **Unable to read PDF.** The file may be corrupted."

    if reader.is_encrypted:
        return "", 0, "🔒 **This PDF is password-protected.** Please decrypt first."

    page_count = len(reader.pages)
    if page_count == 0:
        return "", 0, "📄 **This PDF contains no pages.**"

    all_text = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            all_text.append(txt)

    full_text = "\n".join(all_text).strip()
    if not full_text:
        return "", page_count, (
            "📷 **No extractable text found.** This appears to be a scanned image PDF. "
            "Please use a text-based research PDF."
        )

    return full_text, page_count, None


def build_analysis_prompt(paper_text):
    """Build structured 8-section research paper analysis prompt."""
    return textwrap.dedent(f"""\
    You are an expert academic research paper analyst. A student has uploaded
    a research paper and needs your help understanding it.

    Analyze the following research paper text carefully and produce a structured
    summary. Write in clear, simple English that a university student can easily understand.

    IMPORTANT RULES:
    - Only include information that is ACTUALLY present in the paper.
    - If a specific detail is not mentioned, clearly state "Not mentioned in the paper."
    - Do NOT invent, assume or hallucinate any information.
    - For numerical results, include them ONLY if they are explicitly stated in the paper.

    Provide your analysis in the following format EXACTLY (use these exact headings
    with markdown formatting shown):

    ## 📄 Research Paper Title
    [Identify the actual title of the paper]

    ## 👥 Authors
    [List the authors if available, otherwise state "Not mentioned in the paper."]

    ## 🔍 Problem Statement
    [Explain what problem the researchers are trying to solve. Use simple, clear language.]

    ## ⚙️ Methodology
    [Explain the techniques, algorithms, models or methods used. Describe how the
    researchers approached the problem in an intuitive way.]

    ## 📊 Dataset
    [Identify the dataset(s) used. Mention available details such as size, source
    or type. If no dataset is mentioned, state "Not mentioned in the paper."]

    ## 📈 Results and Findings
    [Explain the main results and achievements. Include specific numerical metrics
    ONLY if they are explicitly stated in the paper.]

    ## ⚠️ Limitations
    [Explain limitations acknowledged by the authors. If not stated, write
    "The authors did not explicitly discuss limitations in this paper."]

    ## 🎯 Five Key Takeaways
    1. [First key takeaway – short and clear]
    2. [Second key takeaway – short and clear]
    3. [Third key takeaway – short and clear]
    4. [Fourth key takeaway – short and clear]
    5. [Fifth key takeaway – short and clear]

    --- BEGIN RESEARCH PAPER TEXT ---

    {paper_text}

    --- END RESEARCH PAPER TEXT ---
    """)


def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def generate_download_text(summary, filename):
    header = (
        f"ResearchMate AI – Research Paper Analysis\n"
        f"Document: {filename}\n"
        + "=" * 55 + "\n"
        + "Generated by ResearchMate AI (Powered by Google Gemini)\n"
        + "=" * 55 + "\n\n"
    )
    return header + summary.replace("**", "")


# =====================================================================
#  SIDEBAR
# =====================================================================

with st.sidebar:
    st.markdown("### 🎓 ResearchMate AI")
    st.markdown("---")

    if st.session_state.analyzed_file_name:
        st.success(f"📄 **Active Paper:**\n{st.session_state.analyzed_file_name}")
    else:
        st.info("☁️ **Cloud AI Ready:** Upload a PDF or click ☁️ to chat!")

    st.markdown("##### 📌 Tips")
    st.markdown(
        "- Upload text-based PDFs (not scanned images)\n"
        f"- Max file size: {MAX_FILE_SIZE_MB} MB\n"
        "- Click the **☁️ Cloud Symbol** anytime to chat with the paper or ask any question!"
    )

    st.markdown("---")
    if st.button("🗑️ Reset All", use_container_width=True):
        st.session_state.analysis_result = None
        st.session_state.analyzed_file_name = None
        st.session_state.analyzed_file_size = None
        st.session_state.paper_text = ""
        st.session_state.cloud_chat_messages = [st.session_state.cloud_chat_messages[0]]
        st.session_state.show_cloud_chat = False
        st.rerun()

    st.markdown(
        "<div style='text-align:center; color:#90a4ae; font-size:0.75rem; margin-top:2rem;'>"
        "B.Tech AI Academic Project<br>Powered by Google Gemini AI"
        "</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
#  HERO SECTION
# =====================================================================

st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-badge">🎓 AI-POWERED ACADEMIC RESEARCH SUITE</div>
        <div class="hero-title">ResearchMate AI</div>
        <div class="hero-subtitle">Understand Research Papers in Minutes</div>
        <div class="hero-tagline">
            Upload any academic PDF to generate a structured 8-part summary, 
            or tap the <strong>☁️ Cloud Symbol</strong> to consult the AI on your paper or any topic in the world.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
#  ☁️ CLOUD SYMBOL ACTION BAR
# =====================================================================

has_paper = bool(st.session_state.paper_text)

col_bar_info, col_bar_btn = st.columns([3, 1.2])

with col_bar_info:
    if has_paper:
        st.markdown(
            f"""
            <div class="cloud-bar">
                <div class="cloud-bar-left">
                    <span class="cloud-icon">☁️</span>
                    <div>
                        <strong>Cloud AI Grounded with PDF:</strong><br>
                        <span style="font-size:0.85rem; color:#455a64;">
                            Ready to answer questions about <strong>{st.session_state.analyzed_file_name}</strong> + all universal knowledge.
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="cloud-bar">
                <div class="cloud-bar-left">
                    <span class="cloud-icon">☁️</span>
                    <div>
                        <strong>Universal Cloud AI Assistant:</strong><br>
                        <span style="font-size:0.85rem; color:#455a64;">
                            Tap the ☁️ symbol to ask anything on any topic, or upload a PDF below.
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_bar_btn:
    btn_label = "☁️ Close Cloud Chat" if st.session_state.show_cloud_chat else "☁️ Tap Cloud to Chat"
    if st.button(btn_label, use_container_width=True, help="Tap to toggle Cloud AI Chat"):
        st.session_state.show_cloud_chat = not st.session_state.show_cloud_chat
        st.rerun()


# =====================================================================
#  ☁️ CLOUD CHATBOT INTERACTIVE PANEL (SHOWN WHEN CLOUD IS TAPPED)
# =====================================================================

if st.session_state.show_cloud_chat:

    st.markdown(
        """
        <div class="cloud-chat-card">
            <div class="cloud-chat-header">
                <div class="cloud-chat-title">
                    <span>☁️</span> Cloud AI Research Assistant
                </div>
                <span style="font-size:0.8rem; background:#ede7f6; color:#5e35b1; padding:0.25rem 0.6rem; border-radius:12px; font-weight:600;">
                    Omniscient AI • Paper Grounded + Universal Knowledge
                </span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick prompt suggestion buttons
    quick_cols = st.columns(4)
    quick_prompts = [
        "📄 Explain Section 3 in simple words" if has_paper else "🧠 Explain Quantum Computing",
        "❓ What are the main findings of this paper?" if has_paper else "🤖 What are Diffusion Models?",
        "💡 Give real-world applications",
        "💻 Write Python implementation code",
    ]

    selected_quick = None
    for idx, qcol in enumerate(quick_cols):
        with qcol:
            if st.button(quick_prompts[idx], key=f"cloud_quick_{idx}", use_container_width=True):
                selected_quick = quick_prompts[idx]

    st.markdown("---")

    # Render Chat History
    for msg in st.session_state.cloud_chat_messages:
        with st.chat_message(msg["role"], avatar="☁️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Double input: both Send button form and st.chat_input for maximum compatibility
    with st.form("cloud_input_form", clear_on_submit=True):
        form_cols = st.columns([5, 1])
        with form_cols[0]:
            form_text = st.text_input(
                "Message Cloud AI:",
                placeholder="Ask about the paper (e.g. 'Explain Section 3') or any topic in the world...",
                label_visibility="collapsed"
            )
        with form_cols[1]:
            form_submit = st.form_submit_button("Send 🚀", use_container_width=True)

    chat_input_val = st.chat_input("Or type here and press Enter...")

    active_query = None
    if selected_quick:
        active_query = selected_quick
    elif form_submit and form_text.strip():
        active_query = form_text.strip()
    elif chat_input_val and chat_input_val.strip():
        active_query = chat_input_val.strip()

    if active_query:
        # 1. Append user message
        st.session_state.cloud_chat_messages.append({"role": "user", "content": active_query})

        client = get_gemini_client()
        if client:
            with st.spinner("☁️ Cloud AI is replying..."):
                paper_context = ""
                if has_paper:
                    paper_context = (
                        f"\n[REFERENCE RESEARCH PAPER CONTENT: '{st.session_state.analyzed_file_name}']\n"
                        f"{st.session_state.paper_text[:60000]}\n"
                        f"[END CONTEXT]\n"
                    )

                history_str = ""
                for past in st.session_state.cloud_chat_messages[-6:-1]:
                    history_str += f"{past['role'].capitalize()}: {past['content']}\n"

                system_instruction = (
                    "You are the Cloud Research Assistant for ResearchMate AI. "
                    "You possess comprehensive, omniscient knowledge across ALL fields of human inquiry: "
                    "computer science, mathematics, quantum physics, biology, medicine, coding, philosophy, and history. "
                    "If research paper context is provided, you reference its exact content accurately to explain concepts, "
                    "methodologies, metrics, and equations, while seamlessly connecting them with broader knowledge and code examples. "
                    "Provide well-structured, clear, student-friendly responses with markdown formatting."
                )

                full_prompt = (
                    f"{paper_context}\n"
                    f"Previous Conversation:\n{history_str}\n"
                    f"Current Question: {active_query}\n"
                )

                ans, used_m, err = query_gemini_robust(client, full_prompt, system_instruction)
                if ans:
                    st.session_state.cloud_chat_messages.append({"role": "assistant", "content": ans})
                else:
                    err_text = f"❌ Error: {err}"
                    st.session_state.cloud_chat_messages.append({"role": "assistant", "content": err_text})

        # Instant rerun so the reply appears immediately
        st.rerun()

    col_cl1, col_cl2 = st.columns([4, 1])
    with col_cl2:
        if st.button("🧹 Clear Cloud Chat", use_container_width=True):
            st.session_state.cloud_chat_messages = [st.session_state.cloud_chat_messages[0]]
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
#  PDF UPLOAD & ANALYSIS SECTION
# =====================================================================

st.markdown("---")

# How it works process pills
st.markdown(
    """
    <div class="steps-row">
        <div class="step-pill">
            <div class="step-num">1</div><br>
            <span class="step-label">📤 Upload PDF</span>
        </div>
        <div class="step-pill">
            <div class="step-num">2</div><br>
            <span class="step-label">📝 Extract Text</span>
        </div>
        <div class="step-pill">
            <div class="step-num">3</div><br>
            <span class="step-label">🤖 Gemini AI</span>
        </div>
        <div class="step-pill">
            <div class="step-num">4</div><br>
            <span class="step-label">📊 8-Part Summary</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a Research Paper (PDF format)",
    type=["pdf"],
    help=f"Upload text-based PDF up to {MAX_FILE_SIZE_MB} MB",
)

if uploaded_file is not None:
    file_size = uploaded_file.size
    file_name = uploaded_file.name

    st.markdown(
        f'<div class="file-badge">'
        f"📄 <strong>{file_name}</strong> &nbsp;|&nbsp; {format_file_size(file_size)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"🚫 File exceeds {MAX_FILE_SIZE_MB} MB limit.")
    else:
        already_analyzed = (
            st.session_state.analysis_result is not None
            and st.session_state.analyzed_file_name == file_name
            and st.session_state.analyzed_file_size == file_size
        )

        col_btn1, col_btn2, col_btn3 = st.columns([3, 1.2, 1])

        with col_btn1:
            analyze_clicked = st.button(
                "🔬 Analyze Research Paper",
                use_container_width=True,
                disabled=already_analyzed,
                key="btn_analyze_main"
            )

        with col_btn2:
            chat_with_paper_clicked = st.button(
                "☁️ Chat with Paper",
                use_container_width=True,
                help="Open Cloud AI Assistant for this paper"
            )
            if chat_with_paper_clicked:
                st.session_state.show_cloud_chat = True
                st.rerun()

        with col_btn3:
            if st.button("🗑️ Clear", use_container_width=True, key="btn_clear_paper"):
                st.session_state.analysis_result = None
                st.session_state.analyzed_file_name = None
                st.session_state.analyzed_file_size = None
                st.session_state.paper_text = ""
                st.session_state.show_cloud_chat = False
                st.rerun()

        if already_analyzed:
            st.info("💡 Paper is already analyzed and loaded in Cloud AI memory. Tap **☁️ Chat with Paper** to ask questions!")

        if analyze_clicked and not already_analyzed:
            client = get_gemini_client()
            if client:
                with st.status("🔄 Processing research paper...", expanded=True) as status:
                    st.write("📄 Extracting text from PDF...")
                    text, page_count, err = extract_text_from_pdf(uploaded_file)

                    if err:
                        status.update(label="❌ Extraction failed", state="error")
                        st.error(err)
                    else:
                        st.session_state.paper_text = text
                        st.write(f"✅ Extracted **{page_count} pages** ({len(text):,} characters)")

                        trunc_text = text[:MAX_CHARS_FOR_API]
                        st.write("🤖 Generating structured 8-part summary with Gemini AI...")

                        prompt = build_analysis_prompt(trunc_text)
                        res, used_model, err_api = query_gemini_robust(client, prompt)

                        if res:
                            st.session_state.analysis_result = res
                            st.session_state.analyzed_file_name = file_name
                            st.session_state.analyzed_file_size = file_size
                            status.update(label=f"✅ Analysis complete using {used_model}!", state="complete")
                        else:
                            status.update(label="❌ Analysis failed", state="error")
                            st.error(f"AI Error: {err_api}")

    # Render Analysis Results
    if st.session_state.analysis_result:
        st.markdown("---")
        st.markdown(
            "<h3 style='text-align:center; color:#1a237e;'>📊 Structured Research Paper Summary</h3>",
            unsafe_allow_html=True,
        )

        summary = st.session_state.analysis_result
        sections = summary.split("\n## ")

        for i, section in enumerate(sections):
            if i == 0 and not summary.startswith("## "):
                if section.strip():
                    st.markdown(section.strip())
                continue

            heading = section.split("\n")[0].strip()
            lines = section.split("\n", 1)
            body = lines[1].strip() if len(lines) > 1 else ""

            with st.expander(heading if heading else "Analysis Section", expanded=True):
                st.markdown(body)

        # Download & Copy actions
        st.markdown("---")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            dl_txt = generate_download_text(summary, st.session_state.analyzed_file_name)
            st.download_button(
                label="📥 Download Full Summary (.txt)",
                data=dl_txt,
                file_name=f"Summary_{st.session_state.analyzed_file_name}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_d2:
            if st.button("📋 Open Copyable Summary View", use_container_width=True):
                st.text_area("Full Summary:", value=summary, height=280)

else:
    st.markdown(
        """
        <div style="background:#ffffff; border:1px solid #e3e8f0; border-radius:12px; text-align:center; padding:2.2rem 1rem; margin-top:1rem;">
            <h4 style="color:#1a237e; margin-bottom:0.4rem;">📄 Ready to Analyze Research Papers</h4>
            <p style="color:#546e7a; max-width:550px; margin:0 auto; font-size:0.92rem;">
                Upload an academic paper in PDF format above to generate an instant 8-part summary, 
                or tap the <strong>☁️ Cloud Symbol</strong> to consult the AI on any topic!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
#  FOOTER
# =====================================================================

st.markdown(
    """
    <div class="app-footer">
        ResearchMate AI • Built with Streamlit &amp; Google Gemini AI<br>
        B.Tech Artificial Intelligence Academic Project • Intelligent Summarizer &amp; Cloud AI
    </div>
    """,
    unsafe_allow_html=True,
)
