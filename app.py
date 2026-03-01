"""
app.py

Purpose:
Streamlit UI for Delaware case study :
- Multi-threaded chat
- Streaming answers
- Retrieval transparency (sources/pages/chunk previews)
- Hybrid RAG + Agentic routing handled in main.py
- Optional DeepEval + Langfuse logging (non-blocking)

Author:
Karan Kadam
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import main  # final Hybrid RAG + Agentic Routing pipeline

# Evals / Langfuse
try:
    from evals import evaluate_and_log, get_langfuse_handler
except Exception:
    evaluate_and_log = None
    get_langfuse_handler = None


# ----------------------------
# Helpers
# ----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@st.cache_resource(show_spinner=False)
def get_vs_and_bm25() -> Tuple[Any, List[Any]]:
    vs = main.load_vectorstore(main.VECTORSTORE_PATH, main.EMBEDDING_MODEL)
    bm25_corpus = main.load_all_chunks_for_bm25(vs)
    return vs, bm25_corpus


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOpenAI:
    return main.init_llm(main.DEFAULT_LLM_MODEL)


def _safe_eval_async(
    *,
    question: str,
    answer: str,
    debug: Dict[str, Any],
    session_id: str,
) -> None:
    """
    Runs DeepEval + Langfuse logging without blocking Streamlit UI.

    IMPORTANT:
    evals.evaluate_and_log signature:
      evaluate_and_log(question, answer, meta, cfg, thread_id, turn_id, trace_id=None)

    - cfg: cfg dict + mode/retrieval_result
    - thread_id: session_id (streamlit thread)
    - turn_id: unique per assistant response
    - trace_id: optional; create one so Langfuse groups the observation
    """
    if evaluate_and_log is None:
        return

    try:
        # thread_id groups a conversation thread
        thread_id = session_id
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        trace_id = str(uuid.uuid4())

        # Build eval cfg similar to CLI flow
        cfg_for_eval = dict((debug.get("cfg") or {}))
        cfg_for_eval["mode"] = debug.get("mode", "rag")
        cfg_for_eval["retrieval_result"] = debug.get("retrieval_result", "unknown")

        # Non-blocking eval; failures should not crash UI
        evaluate_and_log(
            question=question,
            answer=answer,
            meta=debug,
            cfg=cfg_for_eval,
            thread_id=thread_id,
            turn_id=turn_id,
            trace_id=trace_id,
        )
    except Exception:
        # swallow all eval errors in UI mode
        return


# ----------------------------
# Streaming pipeline
# ----------------------------
def run_pipeline_stream(
    question: str,
    *,
    vs: Any,
    llm: ChatOpenAI,
    cfg: main.RetrievalConfig,
    bm25_corpus: List[Any],
) -> Tuple[Iterator[str], Dict[str, Any]]:

    q = (question or "").strip()
    if not q:

        def _empty() -> Iterator[str]:
            yield "Please enter a question."

        return _empty(), {"error": "empty_question"}

    if main._is_smalltalk(q):

        def _small() -> Iterator[str]:
            llm_kwargs = main._llm_kwargs()
            for chunk in llm.stream(
                [
                    {"role": "system", "content": "You are a friendly helpful assistant. Reply naturally and briefly."},
                    {"role": "user", "content": q},
                ],
                **llm_kwargs,
            ):
                token = getattr(chunk, "content", "") or ""
                if token:
                    yield token

        debug = {
            "mode": "smalltalk",
            "retrieval_hint": "none",
            "retrieval_result": "skipped_smalltalk",
            "cfg": cfg.__dict__,
            "retrieval": {"sources": [], "pages": [], "chunks_used": 0, "previews": []},
            "retrieval_context": [],
            "timestamp_utc": _utc_now_iso(),
        }
        return _small(), debug

    route = main.route_query(q, llm)

    vector_docs, lexical_docs = [], []

    if route.retrieval_hint in {"vector", "hybrid"}:
        vector_docs = main.vector_retrieve(q, vs, cfg)

    if route.retrieval_hint in {"lexical", "hybrid"}:
        lexical_docs = main.lexical_retrieve(q, bm25_corpus, top_k=cfg.top_k)

    if route.retrieval_hint == "hybrid":
        chosen_docs = main.hybrid_fuse(vector_docs, lexical_docs, cfg)
    elif route.retrieval_hint == "lexical":
        chosen_docs = lexical_docs[: cfg.top_k]
    else:
        chosen_docs = vector_docs[: cfg.top_k]

    if cfg.use_rerank and chosen_docs:
        chosen_docs = main.rerank_with_llm(q, chosen_docs, top_k=cfg.rerank_top_k, llm=llm)

    context = main.build_context(chosen_docs)
    retrieval_payload = main.build_retrieval_payload(chosen_docs, top_k=cfg.top_k)

    # IMPORTANT: build retrieval_context for eval faithfulness
    retrieval_context, retrieved_tags = main.build_retrieval_context(chosen_docs)

    retrieval_result = "ok" if context.strip() else "empty_context"

    debug = {
        "mode": route.mode,
        "retrieval_hint": route.retrieval_hint,
        "retrieval_result": retrieval_result,
        "cfg": cfg.__dict__,
        "retrieval": retrieval_payload,
        "retrieval_context": retrieval_context,
        "retrieved_tags": retrieved_tags,
        "timestamp_utc": _utc_now_iso(),
    }

    if not context.strip():

        def _nf() -> Iterator[str]:
            yield "Not found in provided documents."

        return _nf(), debug

    effective_question = q
    if route.mode == "summarize":
        effective_question = f"Summarize the relevant parts related to: {q}"
    elif route.mode == "extract_kpi":
        effective_question = f"Extract KPIs/metrics/numbers related to: {q}. List them clearly."

    user_prompt = main.RAG_USER_TEMPLATE.format(context=context, question=effective_question)

    def _gen() -> Iterator[str]:
        for chunk in llm.stream(
            [
                {"role": "system", "content": main.RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        ):
            token = getattr(chunk, "content", "") or ""
            if token:
                yield token

    return _gen(), debug


# ----------------------------
# UI State
# ----------------------------
def _init_state() -> None:
    if "threads" not in st.session_state:
        st.session_state.threads = {}
    if "active_thread" not in st.session_state:
        st.session_state.active_thread = _new_thread()


def _new_thread() -> str:
    tid = f"thread-{uuid.uuid4().hex[:8]}"
    st.session_state.threads[tid] = {
        "name": f"Chat {len(st.session_state.threads) + 1}",
        "messages": [],
    }
    return tid


def _get_thread(tid: str) -> Dict[str, Any]:
    return st.session_state.threads[tid]


# ----------------------------
# Streamlit App
# ----------------------------
def main_app() -> None:
    load_dotenv()

    st.set_page_config(
        page_title="ESG Document Intelligence System",
        page_icon="📄",
        layout="wide",
    )

    st.title("ESG Document Intelligence System - Hybrid RAG + Agentic Routing")
    st.caption("Multi-threaded document Q&A with streaming answers and retrieval transparency.")

    _init_state()

    vs, bm25_corpus = get_vs_and_bm25()
    llm = get_llm()

    # Sidebar: Threads only
    with st.sidebar:
        st.subheader("Threads")

        thread_ids = list(st.session_state.threads.keys())
        active = st.session_state.active_thread

        names = {tid: st.session_state.threads[tid]["name"] for tid in thread_ids}
        selected = st.selectbox(
            "Select thread",
            options=list(names.values()),
            index=list(names.values()).index(names[active]),
        )

        st.session_state.active_thread = [tid for tid, name in names.items() if name == selected][0]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New thread", use_container_width=True):
                st.session_state.active_thread = _new_thread()
                st.rerun()
        with col2:
            if st.button("Clear thread", use_container_width=True):
                t = _get_thread(st.session_state.active_thread)
                t["messages"] = []
                st.rerun()

    # Render chat history (USER + ASSISTANT)
    thread_id = st.session_state.active_thread
    thread = _get_thread(thread_id)

    for m in thread["messages"]:
        role = m.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(m.get("content", ""))
            if role == "assistant" and m.get("debug"):
                with st.expander("Retrieval transparency"):
                    st.json(m["debug"])

    # Input
    prompt = st.chat_input("Ask a question about the ESG / financial report...")
    if not prompt:
        return

    # 1) Append user message FIRST
    thread["messages"].append({"role": "user", "content": prompt})

    # 2) Render the user bubble immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    cfg = main.RetrievalConfig()

    # 3) Stream assistant answer
    with st.chat_message("assistant"):
        full: List[str] = []
        placeholder = st.empty()

        gen, debug = run_pipeline_stream(
            prompt,
            vs=vs,
            llm=llm,
            cfg=cfg,
            bm25_corpus=bm25_corpus,
        )

        for tok in gen:
            full.append(tok)
            placeholder.markdown("".join(full))

        answer_text = "".join(full).strip() or "Not found in provided documents."
        placeholder.markdown(answer_text)

        with st.expander("Retrieval transparency"):
            st.json(debug)

    # 4) Persist assistant message
    thread["messages"].append({"role": "assistant", "content": answer_text, "debug": debug})

    # 5) Trigger eval + langfuse logging (non-blocking)
    if evaluate_and_log is not None:
        t = threading.Thread(
            target=_safe_eval_async,
            kwargs={
                "question": prompt,
                "answer": answer_text,
                "debug": debug,
                "session_id": thread_id,
            },
            daemon=True,
        )
        t.start()

    # 6) Rerun so history re-renders cleanly
    st.rerun()


if __name__ == "__main__":
    main_app()