
import streamlit as st
import requests


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    section[data-testid="stSidebar"] {
        background-color: #20202e;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://api:8000"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_api():
    """Check whether FastAPI is running."""

    try:
        response = requests.get(
            f"{API_URL}/docs",
            timeout=3,
        )

        return response.status_code == 200

    except Exception:
        return False


def ask_question(question):
    """Send question to FastAPI."""

    response = requests.post(
        f"{API_URL}/v1/ask",
        json={
            "question": question
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


def get_value(data, *keys, default=None):

    for key in keys:

        if key in data:
            return data[key]

    return default


def extract_citations(data):

    citations = get_value(
        data,
        "citations",
        "sources",
        "references",
        default=[],
    )

    if citations is None:
        return []

    if isinstance(citations, dict):
        citations = [citations]

    return citations


def calculate_confidence(data):

    confidence = get_value(
        data,
        "confidence",
        "score",
        "quality",
        default=None,
    )

    if confidence is None:
        return None

    try:

        confidence = float(confidence)

        if confidence <= 1:
            confidence *= 100

        return max(
            0,
            min(100, confidence),
        )

    except Exception:

        return None


def get_unique_sources(citations):

    unique_sources = set()

    for citation in citations:

        if isinstance(citation, dict):

            source = citation.get(
                "source",
                citation.get(
                    "filename",
                    "",
                ),
            )

            if source:
                unique_sources.add(
                    str(source)
                )

        else:

            if citation:
                unique_sources.add(
                    str(citation)
                )

    return unique_sources


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ System")

    st.markdown("**FastAPI:**")

    st.code(
    "http://api:8000",
    language="text",
)

    st.markdown("**Ask endpoint:**")

    st.code(
        "/v1/ask",
        language="text",
    )

    st.divider()

    st.markdown(
        "### 🔍 Hybrid retrieval pipeline"
    )

    st.markdown(
        """
        Dense Retrieval → BM25 → RRF →  
        Cross-Encoder → Groq
        """
    )

    st.divider()

    st.markdown("### System status")

    api_online = check_api()

    if api_online:

        st.success("🟢 RAG Ready")

    else:

        st.error("🔴 API Offline")

    st.markdown("**Indexed Chunks**")

    st.markdown("# 3095")

    st.markdown("BM25: 🟢 Ready")

    st.markdown("Reranker: 🟢 Loaded")

    st.divider()

    st.markdown("### Indexed Documents")

    st.markdown("**Documents:** 2")

    with st.expander("View documents"):

        st.write(
            "📄 Building_Machine_Learning_Systems.pdf"
        )

        st.write(
            "📄 Deep_Learning.pdf"
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("📚 Hybrid RAG Assistant")

st.markdown(
    "Ask questions about the documents indexed by the "
    "Hybrid Retrieval-Augmented Generation system."
)


# ============================================================
# API STATUS
# ============================================================

if api_online:

    st.success(
        "🟢 RAG API is online and ready."
    )

else:

    st.error(
        "RAG API is offline. Start FastAPI first."
    )

    st.code(
        "uvicorn app.main:app --reload --port 8000"
    )


# ============================================================
# QUESTION INPUT
# ============================================================

st.header("Ask a question")

question = st.text_area(
    "Enter your question:",
    value="What is machine learning?",
    height=110,
)

ask_button = st.button(
    "🔍 Ask Question",
    use_container_width=True,
    type="primary",
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not api_online:

        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure the FastAPI server is running."
        )

    else:

        with st.spinner(
            "Running hybrid retrieval pipeline..."
        ):

            try:

                data = ask_question(
                    question.strip()
                )

                st.session_state["result"] = data

                st.session_state["question"] = (
                    question.strip()
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. "
                    "The RAG pipeline may still be processing."
                )

            except requests.exceptions.HTTPError as error:

                st.error(
                    f"FastAPI returned an error: {error}"
                )

                if error.response is not None:

                    st.code(
                        error.response.text
                    )

            except Exception as error:

                st.error(
                    f"Could not connect to FastAPI: {error}"
                )


# ============================================================
# DISPLAY RESULT
# ============================================================

if "result" in st.session_state:

    data = st.session_state["result"]

    current_question = st.session_state.get(
        "question",
        question,
    )


    # ========================================================
    # ANSWER
    # ========================================================

    answer = get_value(
        data,
        "answer",
        "response",
        default="No answer returned.",
    )

    st.divider()

    st.header("💡 Answer")

    st.write(answer)


    # ========================================================
    # ANSWER QUALITY
    # ========================================================

    citations = extract_citations(data)

    confidence = calculate_confidence(data)

    unique_sources = get_unique_sources(
        citations
    )

    st.header("📊 Answer Quality")


    # --------------------------------------------------------
    # METRIC BOXES
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        if confidence is not None:

            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%",
            )

        else:

            st.metric(
                label="Confidence",
                value="N/A",
            )


    with col2:

        st.metric(
            label="Citations",
            value=len(citations),
        )


    with col3:

        st.metric(
            label="Sources",
            value=len(unique_sources),
        )


    # ========================================================
    # CONFIDENCE STATUS
    # ========================================================

    if confidence is not None:

        if confidence >= 80:

            st.success(
                f"↑ High confidence — {confidence:.1f}%"
            )

        elif confidence >= 50:

            st.warning(
                f"→ Medium confidence — {confidence:.1f}%"
            )

        else:

            st.error(
                f"↓ Low confidence — {confidence:.1f}%"
            )


    # ========================================================
    # CITATIONS
    # ========================================================

    st.header("📚 Citations")

    if citations:

        for index, citation in enumerate(
            citations,
            start=1,
        ):

            # ------------------------------------------------
            # Each citation gets its own Streamlit box
            # ------------------------------------------------

            with st.container(border=True):

                st.subheader(
                    f"📄 Citation [{index}]"
                )

                if isinstance(
                    citation,
                    dict,
                ):

                    source = citation.get(
                        "source",
                        citation.get(
                            "filename",
                            "Unknown source",
                        ),
                    )

                    page = citation.get(
                        "page",
                        None,
                    )

                    human_page = citation.get(
                        "human_page",
                        None,
                    )

                    reranker_score = citation.get(
                        "reranker_score",
                        None,
                    )


                    # Source

                    st.markdown(
                        f"**Source:** `{source}`"
                    )


                    # Page

                    if page is not None:

                        if human_page is not None:

                            st.markdown(
                                f"**Page:** {page} "
                                f"(Human page: {human_page})"
                            )

                        else:

                            st.markdown(
                                f"**Page:** {page}"
                            )


                    # Reranker score

                    if reranker_score is not None:

                        st.markdown(
                            f"**Reranker score:** "
                            f"{float(reranker_score):.4f}"
                        )

                else:

                    st.write(
                        citation
                    )

    else:

        st.info(
            "No citation information was returned by the API."
        )


    # ========================================================
    # SOURCES
    # ========================================================

    st.header("📁 Sources")

    if unique_sources:

        for source in sorted(unique_sources):

            st.write(
                f"📄 {source}"
            )

    else:

        st.info(
            "No source information was returned by the API."
        )


    # ========================================================
    # QUESTION
    # ========================================================

    st.header("❓ Question")

    st.write(
        current_question
    )