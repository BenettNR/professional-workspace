"""Streamlit dashboard for the RAG Hybrid Search system.

Features:
  - Natural language Q&A with grounded answers
  - Inline citation display with verification status
  - 3-dimensional confidence score breakdown
  - Side-by-side hybrid vs dense-only retrieval comparison
  - Document library view
  - File upload for new documents
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Hybrid Search",
    page_icon="🔍",
    layout="wide",
)


def _is_offline_mode() -> tuple[bool, str, str]:
    """Detect offline mode by re-applying the same auto-detection rules as the API.

    Returns (offline, embedding_backend, llm_backend).
    """
    embedding_backend = os.getenv("EMBEDDING_BACKEND", "").strip()
    llm_backend = os.getenv("LLM_BACKEND", "").strip()
    voyage_key = os.getenv("VOYAGE_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if not embedding_backend:
        embedding_backend = (
            "local" if (not voyage_key or voyage_key.startswith("pa-...")) else "voyage"
        )
    if not llm_backend:
        llm_backend = (
            "replay"
            if (not anthropic_key or anthropic_key.startswith("sk-ant-..."))
            else "anthropic"
        )

    offline = embedding_backend == "local" or llm_backend == "replay"
    return offline, embedding_backend, llm_backend


def _load_demo_questions() -> list[str]:
    path = Path(os.getenv("DEMO_QUESTIONS_PATH", "eval/demo_questions.json"))
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [q["question"] for q in data.get("questions", [])]
    except (OSError, json.JSONDecodeError):
        return []


_OFFLINE, _EMB_BACKEND, _LLM_BACKEND = _is_offline_mode()
if _OFFLINE:
    st.warning(
        f"**🟡 Demo mode (offline)** — `embedding_backend={_EMB_BACKEND}`, "
        f"`llm_backend={_LLM_BACKEND}`. Answers are replayed from a recorded "
        f"fixture set. Set `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY` in `.env` "
        f"for live mode."
    )
    _DEMO_QUESTIONS = _load_demo_questions()
    if _DEMO_QUESTIONS:
        with st.expander(f"Available demo questions ({len(_DEMO_QUESTIONS)})", expanded=False):
            for q in _DEMO_QUESTIONS:
                st.write(f"- {q}")


# ── helpers ───────────────────────────────────────────────────────────────────


def _api(method: str, path: str, **kwargs) -> dict | None:
    try:
        with httpx.Client(timeout=120) as client:
            response = getattr(client, method)(f"{API_BASE}{path}", **kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")
    return None


def _confidence_gauge(label: str, value: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            title={"text": label, "font": {"size": 12}},
            number={"suffix": "%", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 40], "color": "#ffcccc"},
                    {"range": [40, 70], "color": "#fff0cc"},
                    {"range": [70, 100], "color": "#ccffcc"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": 30,
                },
            },
        )
    )
    fig.update_layout(height=200, margin={"t": 40, "b": 0, "l": 10, "r": 10})
    return fig


def _citation_badge(verified: bool) -> str:
    return "✅ Verified" if verified else "⚠️ Unverified"


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("RAG Hybrid Search")
    st.caption("Production-grade RAG with hybrid dense+sparse retrieval")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Ask a Question", "Document Library", "Upload Document"],
        label_visibility="collapsed",
    )

    st.divider()
    dense_only = st.toggle(
        "Dense-only mode",
        value=False,
        help="Disable BM25 sparse retrieval for ablation comparison",
    )
    st.caption("Toggle to compare hybrid vs dense-only retrieval")


# ── Ask a Question ────────────────────────────────────────────────────────────

if page == "Ask a Question":
    st.header("Ask a Question")
    st.caption(
        "Hybrid retrieval combines semantic vector search with BM25 keyword matching — "
        "ideal for technical documentation with exact function names and config keys."
    )

    with st.form("ask_form"):
        question = st.text_area(
            "Your question",
            placeholder="e.g. What are the rate limits for the Nexus API?",
            height=100,
        )
        col1, col2 = st.columns([1, 5])
        submitted = col1.form_submit_button("Ask", type="primary", use_container_width=True)
        if dense_only:
            col2.info("Dense-only mode active — BM25 disabled")

    if submitted and question.strip():
        with st.spinner("Retrieving and generating…"):
            result = _api(
                "post",
                "/v1/ask",
                json={"question": question, "dense_only": dense_only},
            )

        if result:
            # ── Answer ─────────────────────────────────────────────────────
            st.subheader("Answer")
            if result.get("insufficient_info"):
                st.warning(result.get("missing_info_message", "Insufficient context."))

            st.markdown(result["answer"])

            # ── Confidence ─────────────────────────────────────────────────
            st.subheader("Confidence Breakdown")
            conf = result["confidence"]
            c1, c2, c3, c4 = st.columns(4)
            c1.plotly_chart(
                _confidence_gauge("Retrieval", conf["retrieval_confidence"], "#4A90D9"),
                use_container_width=True,
            )
            c2.plotly_chart(
                _confidence_gauge("Citations", conf["citation_coverage"], "#7BC67E"),
                use_container_width=True,
            )
            c3.plotly_chart(
                _confidence_gauge("Completeness", conf["answer_completeness"], "#F0A500"),
                use_container_width=True,
            )
            c4.plotly_chart(
                _confidence_gauge("Composite", conf["composite"], "#7B2D8B"),
                use_container_width=True,
            )

            # ── Citations ──────────────────────────────────────────────────
            citations = result.get("citations", [])
            if citations:
                st.subheader(f"Citations ({len(citations)})")
                for c in citations:
                    badge = _citation_badge(c["verified"])
                    with st.expander(
                        f"[{c['number']}] {c['source_file'].split('/')[-1]}  {badge}",
                        expanded=False,
                    ):
                        st.markdown(f"**Source:** `{c['source_file']}`")
                        st.markdown(f"**Status:** {badge}")
                        st.markdown(f"**Reason:** {c['verification_reason']}")
                        st.markdown(f"**Excerpt:** *{c['text_excerpt']}…*")

            # ── Retrieved Chunks ───────────────────────────────────────────
            chunks = result.get("retrieved_chunks", [])
            if chunks:
                st.subheader(f"Retrieved Context ({len(chunks)} chunks)")
                mode_label = "dense-only" if dense_only else "hybrid (dense + sparse)"
                st.caption(f"Retrieval mode: {mode_label}")
                for i, chunk in enumerate(chunks, 1):
                    scores = []
                    if chunk.get("rerank_score") is not None:
                        scores.append(f"rerank={chunk['rerank_score']:.3f}")
                    if chunk.get("fusion_score"):
                        scores.append(f"fusion={chunk['fusion_score']:.4f}")
                    if chunk.get("dense_score") is not None:
                        scores.append(f"dense={chunk['dense_score']:.3f}")
                    if chunk.get("sparse_score") is not None:
                        scores.append(f"sparse={chunk['sparse_score']:.3f}")
                    score_str = "  |  ".join(scores)
                    with st.expander(
                        f"#{i} {chunk['filename']} — chunk {chunk['chunk_index']}  ({score_str})",
                        expanded=(i == 1),
                    ):
                        st.text(chunk["content"])


# ── Document Library ──────────────────────────────────────────────────────────

elif page == "Document Library":
    st.header("Indexed Documents")
    result = _api("get", "/v1/documents")
    if result:
        st.metric("Total Documents", result["total_documents"])
        if result["documents"]:
            import pandas as pd

            df = pd.DataFrame(result["documents"])
            st.dataframe(
                df[["filename", "chunking_strategy", "total_chunks", "source_file"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No documents indexed yet. Use the Upload tab to add documents.")


# ── Upload Document ────────────────────────────────────────────────────────────

elif page == "Upload Document":
    st.header("Upload Document")
    st.caption("Supported formats: Markdown, plain text, HTML, PDF")

    with st.form("upload_form"):
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["md", "txt", "html", "htm", "pdf"],
        )
        strategy = st.selectbox(
            "Chunking strategy",
            ["recursive", "fixed", "semantic"],
            help=(
                "recursive: splits on section headers (recommended)\n"
                "fixed: character-count windows\n"
                "semantic: topic-boundary detection via embeddings (slow)"
            ),
        )
        submitted = st.form_submit_button("Index Document", type="primary")

    if submitted and uploaded_file:
        with st.spinner(f"Indexing '{uploaded_file.name}'…"):
            result = _api(
                "post",
                "/v1/ingest",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                data={"strategy": strategy},
            )
        if result:
            st.success(result["message"])
            col1, col2 = st.columns(2)
            col1.metric("Chunks Indexed", result["chunks_indexed"])
            col2.metric("Duplicates Skipped", result["chunks_skipped"])
