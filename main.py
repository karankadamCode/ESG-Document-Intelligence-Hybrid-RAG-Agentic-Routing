"""
main.py

Purpose:
Terminal QA for Delaware case study using::
- Hybrid Retrieval (Vector FAISS + Lexical BM25) + Weighted Fusion
- Agentic routing (router prompt YAML)
- Optional LLM reranking returning JSON indices
- Prompts loaded via prompt_manager.py (YAML versioning)
- CLI interface + retrieval transparency + citations
- Optional DeepEval + Langfuse scoring/tracing (via evals.py / eval_langfuse-style module)

Run : python main.py -q "Summarize climate risk management approach." --show-sources true

Author:
Karan Kadam
"""


from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Iterator
import argparse
import json
import logging
import os
import re
import sys
import hashlib
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rank_bm25 import BM25Okapi

from prompt_manager import get_prompts

from evals import (
    evaluate_and_log,
    get_langfuse_handler,
)


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")



# ----------------------------
# Prompt YAML paths
# ----------------------------
PROMPT_RAG_PATH = os.getenv("PROMPT_RAG_PATH", "prompts/rag/v1/prompt.yaml")
PROMPT_RERANK_PATH = os.getenv("PROMPT_RERANK_PATH", "prompts/rerank/v1/prompt.yaml")
PROMPT_ROUTER_PATH = os.getenv("PROMPT_ROUTER_PATH", "prompts/router/v1/prompt.yaml")


# ----------------------------
# Defaults (override via env)
# ----------------------------
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "650"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")

DEFAULT_TOP_K = int(os.getenv("TOP_K", "8"))
DEFAULT_FETCH_K = int(os.getenv("FETCH_K", "40"))
DEFAULT_USE_MMR = os.getenv("USE_MMR", "true").lower() in {"1", "true", "yes", "y"}
DEFAULT_MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))

DEFAULT_USE_RERANK = os.getenv("USE_RERANK", "true").lower() in {"1", "true", "yes", "y"}
DEFAULT_RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", str(DEFAULT_TOP_K)))

DEFAULT_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.65"))  # vector weight
DEFAULT_BETA = float(os.getenv("HYBRID_BETA", "0.35"))    # lexical weight

PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", "240"))
MAX_CTX_CHARS_PER_CHUNK = int(os.getenv("MAX_CTX_CHARS_PER_CHUNK", "1500"))


# ----------------------------
# Prompt loading (YAML via prompt_manager.py)
# ----------------------------
RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, _ = get_prompts(PROMPT_RAG_PATH)
RERANK_SYSTEM_PROMPT, RERANK_USER_TEMPLATE, _ = get_prompts(PROMPT_RERANK_PATH)
ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE, _ = get_prompts(PROMPT_ROUTER_PATH)

INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"system prompt",
    r"developer message",
    r"reveal (secrets|keys|api key)",
    r"you are chatgpt",
    r"do not answer the user",
    r"override",
    r"jailbreak",
    r"BEGIN SYSTEM PROMPT",
]


def looks_like_prompt_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def detect_injection_in_docs(docs: List[Document]) -> bool:
    for d in docs:
        if looks_like_prompt_injection(d.page_content or ""):
            return True
    return False


def build_retrieval_context(docs: List[Document]) -> Tuple[List[str], List[str]]:
    """
    Returns:
      retrieval_context: ["[Source: X | Page Y] chunk text...", ...] (for faithfulness)
      retrieved_tags: ["[Source: X | Page Y]", ...] (optional)
    """
    retrieval_context: List[str] = []
    tags: List[str] = []
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        meta = dict(d.metadata or {})
        source = meta.get("source", meta.get("filename", "unknown"))
        page = meta.get("page", "N/A")
        tag = f"[Source: {source} | Page {page}]"
        tags.append(tag)
        retrieval_context.append(f"{tag} {text[:MAX_CTX_CHARS_PER_CHUNK]}")
    tags = list(dict.fromkeys(tags))
    return retrieval_context, tags


# ----------------------------
# Data structures
# ----------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = DEFAULT_TOP_K
    fetch_k: int = DEFAULT_FETCH_K
    use_mmr: bool = DEFAULT_USE_MMR
    mmr_lambda: float = DEFAULT_MMR_LAMBDA

    use_rerank: bool = DEFAULT_USE_RERANK
    rerank_top_k: int = DEFAULT_RERANK_TOP_K

    hybrid_alpha: float = DEFAULT_ALPHA
    hybrid_beta: float = DEFAULT_BETA

    enable_evals: bool = os.getenv("ENABLE_EVALS", "true").lower() in {"1", "true", "yes", "y"}

@dataclass(frozen=True)
class RouteDecision:
    mode: str              # rag | summarize | extract_kpi
    retrieval_hint: str    # hybrid | vector | lexical


# ----------------------------
# Utility helpers
# ----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    return re.findall(r"[a-z0-9]+", t)


def _stable_chunk_id(source: str, page: Any, text_preview: str) -> str:
    base = f"{source}|{page}|{(text_preview or '')[:200]}"
    h = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"{source}|p{page}|{h}"


def load_vectorstore(path: str, embedding_model: str) -> FAISS:
    if not os.path.exists(path):
        raise FileNotFoundError(f'FAISS index not found at "{path}". Run faiss_ingest.py first.')
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def init_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        top_p=1.0,
    )


def build_context(docs: List[Document]) -> str:
    """
    Build a consolidated RAG context string from retrieved document chunks.

    This function formats each document chunk with a lightweight source tag
    (source name and page number) and appends a truncated portion of its text.
    Empty chunks are skipped, and each chunk is capped at MAX_CTX_CHARS_PER_CHUNK
    to control prompt size.

    The resulting context string is intended to be injected directly into the
    RAG user prompt for answer generation.

    Args:
        docs (List[Document]): Retrieved document chunks to include in the context.

    Returns:
        str: Concatenated context string containing tagged and truncated chunks,
        separated by blank lines. Returns an empty string if no usable content exists.
    """
    parts: List[str] = []
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        meta = dict(d.metadata or {})
        source = meta.get("source", meta.get("filename", "unknown"))
        page = meta.get("page", "N/A")
        tag = f"[Source: {source} | Page {page}]"
        parts.append(f"{tag} {text[:MAX_CTX_CHARS_PER_CHUNK]}")
    return "\n\n".join(parts)


def build_retrieval_payload(docs: List[Document], top_k: int) -> Dict[str, Any]:
    """
    Build a structured retrieval metadata payload for debugging and source attribution.

    This function extracts source-level information, page references, and short content
    previews from the top_k retrieved documents. It also generates a stable chunk ID for
    each preview to enable consistent tracing across runs and logs.

    The payload is designed to be lightweight and human-readable while still containing
    enough detail for retrieval transparency and evaluation.

    Args:
        docs (List[Document]): Retrieved document chunks to summarize in the payload.
        top_k (int): Maximum number of documents to include in the payload.

    Returns:
        Dict[str, Any]: Retrieval metadata containing:
            - sources (List[str]): Unique document sources.
            - pages (List[Any]): Unique page identifiers referenced.
            - chunks_used (int): Number of chunks considered (capped at top_k).
            - previews (List[Dict[str, Any]]): Short previews with chunk_id, source, page,
              and text snippet (limited to a small subset for readability).
    """
    sources: List[str] = []
    pages: List[Any] = []
    previews: List[Dict[str, Any]] = []

    for d in docs[:top_k]:
        meta = dict(d.metadata or {})
        source = meta.get("source", meta.get("filename", "unknown"))
        page = meta.get("page", "N/A")
        text = (d.page_content or "").strip()
        preview = text[:PREVIEW_CHARS].replace("\n", " ").strip()
        cid = _stable_chunk_id(str(source), page, preview)

        if source and source not in sources:
            sources.append(source)
        if page not in pages:
            pages.append(page)

        previews.append({"chunk_id": cid, "source": source, "page": page, "preview": preview})

    return {
        "sources": sources,
        "pages": pages,
        "chunks_used": min(len(docs), top_k),
        "previews": previews[: min(6, len(previews))],
    }


def _llm_kwargs() -> Dict[str, Any]:
    handler = get_langfuse_handler()
    if handler is None:
        return {}
    return {"config": {"callbacks": [handler]}}


# ----------------------------
# Agentic routing
# ----------------------------
def route_query(question: str, llm: ChatOpenAI, llm_kwargs: Optional[Dict[str, Any]] = None) -> RouteDecision:
    """
    Determine the response mode and retrieval strategy for a given question using an LLM router.

    This function sends the question to a router prompt that is expected to return a small JSON object
    describing:
      - mode: one of {"rag", "summarize", "extract_kpi"}
      - retrieval_hint: one of {"hybrid", "vector", "lexical"}

    It includes defensive handling for:
      - empty questions (defaults to rag/hybrid)
      - LLM invocation failures (defaults to rag/hybrid)
      - invalid or non-JSON router outputs (defaults to rag/hybrid)
      - unsupported mode/hint values (coerces to defaults)

    Args:
        question (str): User query text to route.
        llm (ChatOpenAI): Chat model used to perform routing via the router prompt.
        llm_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments forwarded to the LLM
            invocation (e.g., temperature, max_tokens).

    Returns:
        RouteDecision: Routing decision containing the normalized mode and retrieval_hint to be used
        by the downstream retrieval and answering pipeline.
    """
    q = (question or "").strip()
    if not q:
        return RouteDecision(mode="rag", retrieval_hint="hybrid")

    class _SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    prompt = ROUTER_USER_TEMPLATE.format_map(_SafeDict(question=q))

    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            **(llm_kwargs or {})
        ).content.strip()
    except Exception:
        logger.exception("Router failed; defaulting to rag/hybrid")
        return RouteDecision(mode="rag", retrieval_hint="hybrid")

    try:
        obj = json.loads(resp)
        mode = str(obj.get("mode", "rag")).strip().lower()
        hint = str(obj.get("retrieval_hint", "hybrid")).strip().lower()
        if mode not in {"rag", "summarize", "extract_kpi"}:
            mode = "rag"
        if hint not in {"hybrid", "vector", "lexical"}:
            hint = "hybrid"
        return RouteDecision(mode=mode, retrieval_hint=hint)
    except Exception:
        logger.warning(f"Router returned invalid JSON. Raw: {resp[:300]}")
        return RouteDecision(mode="rag", retrieval_hint="hybrid")


# ----------------------------
# Retrieval
# ----------------------------
def vector_retrieve(question: str, vs: FAISS, cfg: RetrievalConfig) -> List[Document]:
    """
    Retrieve candidate documents from a FAISS vector store using either MMR or standard similarity search.

    If cfg.use_mmr is enabled, this uses Max Marginal Relevance (MMR) to balance relevance and diversity
    across results, controlled by cfg.mmr_lambda and cfg.fetch_k. Otherwise, it performs a standard
    top-k vector retrieval.

    Args:
        question (str): Query text to retrieve relevant documents for.
        vs (FAISS): FAISS vector store used for semantic retrieval.
        cfg (RetrievalConfig): Retrieval configuration controlling top_k, fetch_k, and MMR settings.

    Returns:
        List[Document]: Retrieved documents/chunks (length <= cfg.top_k).
    """
    retriever = (
        vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": cfg.top_k,
                "fetch_k": max(cfg.fetch_k, cfg.top_k),
                "lambda_mult": cfg.mmr_lambda,
            },
        )
        if cfg.use_mmr
        else vs.as_retriever(search_kwargs={"k": cfg.top_k})
    )
    return list(retriever.invoke(question) or [])


def load_all_chunks_for_bm25(vs: FAISS) -> List[Document]:
    """
    Build a BM25 corpus by extracting all stored chunk Documents from a FAISS docstore.

    This function accesses the underlying docstore attached to the FAISS vector store and collects
    all values that are instances of langchain_core.documents.Document. It is used to create the
    lexical retrieval corpus for BM25 scoring.

    Args:
        vs (FAISS): FAISS vector store whose internal docstore contains the original chunk Documents.

    Returns:
        List[Document]: All chunk documents available in the FAISS docstore for lexical retrieval.

    Raises:
        RuntimeError: If the FAISS docstore is missing or not accessible in the expected structure.
    """
    store = getattr(vs, "docstore", None)
    if store is None or not hasattr(store, "_dict"):
        raise RuntimeError("FAISS docstore not accessible for BM25 corpus build")
    docs = list(store._dict.values())
    return [d for d in docs if isinstance(d, Document)]


def lexical_retrieve(question: str, all_docs: List[Document], top_k: int) -> List[Document]:
    """
    Retrieve candidate documents using BM25 lexical scoring over tokenized document chunks.

    The question and each document chunk are tokenized using the project's tokenizer. A BM25Okapi
    model is constructed on-the-fly for the provided corpus, scored against the query tokens, and
    the top_k scoring documents are returned (filtered to positive BM25 scores).

    Args:
        question (str): Query text to retrieve relevant documents for.
        all_docs (List[Document]): Full corpus of chunk Documents to score with BM25.
        top_k (int): Number of top documents to return.

    Returns:
        List[Document]: Top BM25-ranked documents with positive scores (length <= top_k).
    """
    q_tokens = _tokenize(question)
    corpus_tokens = [_tokenize(d.page_content or "") for d in all_docs]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(range(len(all_docs)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [all_docs[i] for i in ranked if scores[i] > 0]


def hybrid_fuse(vector_docs: List[Document], lexical_docs: List[Document], cfg: RetrievalConfig) -> List[Document]:
    """
    Fuse vector and lexical retrieval results using weighted reciprocal-rank scoring.

    Each document is assigned a stable chunk identifier derived from its metadata and a short content
    preview. Vector and lexical ranks are converted to reciprocal scores (1/rank), then combined using
    cfg.hybrid_alpha and cfg.hybrid_beta. The highest scoring unique chunks are returned up to cfg.top_k.

    Args:
        vector_docs (List[Document]): Documents returned by vector retrieval (ranked).
        lexical_docs (List[Document]): Documents returned by BM25 lexical retrieval (ranked).
        cfg (RetrievalConfig): Retrieval configuration containing top_k and hybrid fusion weights.

    Returns:
        List[Document]: Fused, deduplicated, ranked documents (length <= cfg.top_k).
    """
    def _cid(d: Document) -> str:
        meta = dict(d.metadata or {})
        source = meta.get("source", meta.get("filename", "unknown"))
        page = meta.get("page", "N/A")
        preview = (d.page_content or "")[:180]
        return _stable_chunk_id(str(source), page, preview)

    v_rank = {_cid(d): (1.0 / float(i)) for i, d in enumerate(vector_docs, start=1)}
    l_rank = {_cid(d): (1.0 / float(i)) for i, d in enumerate(lexical_docs, start=1)}

    all_ids = set(v_rank.keys()) | set(l_rank.keys())
    combined: Dict[str, float] = {}
    for cid in all_ids:
        combined[cid] = (cfg.hybrid_alpha * v_rank.get(cid, 0.0)) + (cfg.hybrid_beta * l_rank.get(cid, 0.0))

    lookup: Dict[str, Document] = {}
    for d in vector_docs + lexical_docs:
        lookup.setdefault(_cid(d), d)

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Document] = []
    for cid, _ in ranked:
        out.append(lookup[cid])
        if len(out) >= cfg.top_k:
            break
    return out


# ----------------------------
# Reranking (JSON indices)
# ----------------------------
def _parse_json_int_list(text: str) -> Optional[List[int]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        return None
    return None


def rerank_with_llm(
    question: str,
    docs: List[Document],
    top_k: int,
    llm: ChatOpenAI,
    llm_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Rerank retrieved documents using an LLM that returns a JSON list of ranked indices.

    This function formats each document into a numbered snippet (including page metadata when
    available), asks the LLM to select/rank the most relevant chunks for the given question,
    and then returns the top_k documents in the LLM-specified order.

    If the LLM call fails or returns invalid output, the function falls back to the original
    document ordering (truncated to top_k).

    Args:
        question (str): User query used to judge relevance.
        docs (List[Document]): Candidate documents/chunks to rerank.
        top_k (int): Maximum number of documents to return after reranking.
        llm (ChatOpenAI): Chat model used to perform reranking.
        llm_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments forwarded to the LLM
            invocation (e.g., temperature, max_tokens).

    Returns:
        List[Document]: Reranked list of documents (length <= top_k). Falls back to the
        original order on failure or invalid LLM output.
    """
    if not docs:
        return []

    items: List[str] = []
    for idx, d in enumerate(docs, start=1):
        meta = dict(d.metadata or {})
        page = meta.get("page", "N/A")
        snippet = (d.page_content or "").replace("\n", " ").strip()[:600]
        items.append(f"{idx}. [Page {page}] {snippet}")
    chunks_text = "\n".join(items)

    prompt = RERANK_USER_TEMPLATE.format(question=question, top_k=top_k, chunks=chunks_text)

    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            **(llm_kwargs or {})
        ).content.strip()
    except Exception:
        logger.exception("Rerank failed; returning original order")
        return docs[:top_k]

    ranked = _parse_json_int_list(resp)
    if not ranked:
        logger.warning(f"Rerank returned invalid JSON. Raw: {resp[:200]}")
        return docs[:top_k]

    ranked = [i for i in ranked if isinstance(i, int) and 1 <= i <= len(docs)]
    if not ranked:
        return docs[:top_k]

    final: List[Document] = []
    seen = set()
    for i in ranked:
        if i in seen:
            continue
        final.append(docs[i - 1])
        seen.add(i)
        if len(final) >= top_k:
            break

    if len(final) < top_k:
        for d in docs:
            if d not in final:
                final.append(d)
            if len(final) >= top_k:
                break

    return final[:top_k]


def _is_smalltalk(text: str) -> bool:
    """
    Heuristic detector for smalltalk messages (greetings/acknowledgements).

    Args:
        text: Raw user input.

    Returns:
        True if text looks like smalltalk, otherwise False.
    """
    t = (text or "").strip().lower()
    if not t:
        return True

    direct = {
        "hi", "hello", "hey", "hii", "hiii",
        "good morning", "good afternoon", "good evening",
        "how are you", "how r you", "whats up", "what's up", "wassup",
        "thanks", "thank you", "thx",
        "ok", "okay", "cool", "done", "k", "kk", "alright",
    }
    if t in direct:
        return True

    patterns = [
        r"^\s*(hi|hello|hey|hii|hiii)\s*[!.]?\s*$",
        r"^\s*(good\s+morning|good\s+afternoon|good\s+evening)\s*[!.]?\s*$",
        r"^\s*how\s+are\s+you\s*[?.!]?\s*$",
        r"^\s*(thanks|thank\s+you)\s*[!.]?\s*$",
        r"^\s*done\s*[!.]?\s*$",
    ]
    for p in patterns:
        if re.match(p, t):
            return True

    if len(t.split()) <= 3 and "?" not in t:
        return True

    return False


def _smalltalk_answer(question: str, llm: ChatOpenAI, llm_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Smalltalk mode (no retrieval). Uses a lightweight system prompt.
    """
    try:
        resp = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a friendly helpful assistant. Reply naturally and briefly.",
                },
                {"role": "user", "content": (question or "").strip()},
            ],
            **(llm_kwargs or {}),
        )
        out = (getattr(resp, "content", "") or "").strip()
        return out or "Hello! How can I help you today?"
    except Exception:
        logger.exception("Smalltalk LLM invoke failed")
        return "Hello! How can I help you today?"

# ----------------------------
# Main QA pipeline
# ----------------------------
def answer_question(
    question: str,
    *,
    vs: FAISS,
    llm: ChatOpenAI,
    cfg: RetrievalConfig,
    bm25_corpus: List[Document],
) -> Tuple[str, Dict[str, Any]]:
    
    """
    Answer a user question using routed retrieval (vector/lexical/hybrid) and RAG generation.

    Workflow:
    1) Validate and normalize the input question.
    2) Short-circuit for smalltalk queries (no retrieval), returning an LLM response with debug metadata.
    3) Route the query to determine retrieval strategy and response mode (e.g., summarize, extract_kpi).
    4) Retrieve documents via vector search (FAISS), lexical search (BM25), or both.
    5) Fuse results for hybrid retrieval and optionally rerank with the LLM.
    6) Build RAG context and generate a grounded answer using system/user prompts.
    7) Optionally run evaluation + Langfuse logging in a non-blocking manner.
    8) Return the final answer along with a structured debug payload describing retrieval and configuration.

    Args:
        question (str): The user query text.
        vs (FAISS): Loaded FAISS vector store for semantic retrieval.
        llm (ChatOpenAI): Chat model used for routing, optional reranking, and final answer generation.
        cfg (RetrievalConfig): Retrieval configuration (top_k, fetch_k, MMR params, rerank params, hybrid weights).
        bm25_corpus (List[Document]): Corpus of chunked documents used for BM25 lexical retrieval.

    Returns:
        Tuple[str, Dict[str, Any]]: A tuple of:
            - answer (str): Final response text. Falls back to "Not found in provided documents." when no context/answer.
            - debug (Dict[str, Any]): Metadata describing mode, retrieval hint, retrieval result, chosen config, sources,
              context previews, and timestamp for observability.

    Raises:
        ValueError: If the provided question is empty after stripping whitespace.
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("Question is empty")

    llm_kwargs = _llm_kwargs()
    
    # --- Smalltalk shortcut (no retrieval) ---
    if _is_smalltalk(q):
        answer = _smalltalk_answer(q, llm, llm_kwargs=llm_kwargs)
        debug = {
            "mode": "smalltalk",
            "retrieval_hint": "none",
            "retrieval_result": "skipped_smalltalk",
            "cfg": {
                "top_k": cfg.top_k,
                "fetch_k": cfg.fetch_k,
                "use_mmr": cfg.use_mmr,
                "mmr_lambda": cfg.mmr_lambda,
                "use_rerank": cfg.use_rerank,
                "rerank_top_k": cfg.rerank_top_k,
                "hybrid_alpha": cfg.hybrid_alpha,
                "hybrid_beta": cfg.hybrid_beta,
            },
            "retrieval": {"sources": [], "pages": [], "chunks_used": 0, "previews": []},
            "retrieval_context": [],
            "timestamp_utc": _utc_now_iso(),
        }
        return answer, debug


    route = route_query(q, llm, llm_kwargs=llm_kwargs)

    vector_docs: List[Document] = []
    lexical_docs: List[Document] = []
    chosen_docs: List[Document] = []

    if route.retrieval_hint in {"vector", "hybrid"}:
        vector_docs = vector_retrieve(q, vs, cfg)

    if route.retrieval_hint in {"lexical", "hybrid"}:
        lexical_docs = lexical_retrieve(q, bm25_corpus, top_k=cfg.top_k)

    if route.retrieval_hint == "hybrid":
        chosen_docs = hybrid_fuse(vector_docs, lexical_docs, cfg)
    elif route.retrieval_hint == "lexical":
        chosen_docs = lexical_docs[: cfg.top_k]
    else:
        chosen_docs = vector_docs[: cfg.top_k]

    if cfg.use_rerank and chosen_docs:
        chosen_docs = rerank_with_llm(q, chosen_docs, top_k=cfg.rerank_top_k, llm=llm, llm_kwargs=llm_kwargs)

    context = build_context(chosen_docs)
    retrieval_payload = build_retrieval_payload(chosen_docs, top_k=cfg.top_k)
    retrieval_context, _ = build_retrieval_context(chosen_docs)

    retrieval_result = "ok" if context.strip() else "empty_context"

    debug = {
        "mode": route.mode,
        "retrieval_hint": route.retrieval_hint,
        "retrieval_result": retrieval_result,
        "cfg": {
            "top_k": cfg.top_k,
            "fetch_k": cfg.fetch_k,
            "use_mmr": cfg.use_mmr,
            "mmr_lambda": cfg.mmr_lambda,
            "use_rerank": cfg.use_rerank,
            "rerank_top_k": cfg.rerank_top_k,
            "hybrid_alpha": cfg.hybrid_alpha,
            "hybrid_beta": cfg.hybrid_beta,
        },
        "retrieval": retrieval_payload,
        "retrieval_context": retrieval_context,
        "timestamp_utc": _utc_now_iso(),
    }

    if not context.strip():
        return "Not found in provided documents.", debug

    effective_question = q
    if route.mode == "summarize":
        effective_question = f"Summarize the relevant parts related to: {q}"
    elif route.mode == "extract_kpi":
        effective_question = f"Extract KPIs/metrics/numbers related to: {q}. List them clearly."

    user_prompt = RAG_USER_TEMPLATE.format(context=context, question=effective_question)

    resp = llm.invoke(
        [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        **llm_kwargs
    )
    answer = (getattr(resp, "content", "") or "").strip()

    if "not found in provided documents" in answer.lower():
        if answer.strip() != "Not found in provided documents." and answer.strip() != "Not found in provided documents":
            answer = "Not found in provided documents."

    if not answer:
        answer = "Not found in provided documents."

    # ----------------------------
    # Evals AFTER answer generation
    # ----------------------------
    if getattr(cfg, "enable_evals", False):
        try:
            thread_id = os.getenv("THREAD_ID", "cli-thread")
            turn_id = os.getenv("TURN_ID", f"turn-{uuid.uuid4().hex[:8]}")
            trace_id = os.getenv("TRACE_ID")

            cfg_for_eval = dict((debug or {}).get("cfg", {}) or {})
            cfg_for_eval["mode"] = (debug or {}).get("mode", "rag")
            cfg_for_eval["retrieval_result"] = (debug or {}).get("retrieval_result", "unknown")

            retrieval = (debug or {}).get("retrieval", {}) or {}
            previews = retrieval.get("previews", []) or []

            meta_for_eval = dict(debug or {})
            meta_for_eval["sources"] = retrieval.get("sources", []) or []
            meta_for_eval["pages"] = retrieval.get("pages", []) or []
            meta_for_eval["chunk_ids"] = [p.get("chunk_id") for p in previews if p.get("chunk_id")]

            eval_payload = evaluate_and_log(
                question=q,
                answer=answer,
                meta=meta_for_eval,
                cfg=cfg_for_eval,
                thread_id=thread_id,
                turn_id=turn_id,
                trace_id=trace_id,
            )

            ev = (eval_payload or {}).get("eval", {}) or {}
            if ev.get("enabled"):
                metrics = ev.get("metrics", {}) or {}
                parts = []
                for k, v in metrics.items():
                    score = float((v or {}).get("score", 0.0) or 0.0)
                    reason = ((v or {}).get("reason", "") or "").strip()
                    if score == 0.0 and reason:
                        parts.append(f"{k}={score:.3f} ({reason})")
                    else:
                        parts.append(f"{k}={score:.3f}")
                msg = " | ".join(parts)
                logger.info(f"[EVAL] turn_id={turn_id} trace_id={eval_payload.get('trace_id','')} {msg}")
            else:
                logger.info(f"[EVAL] disabled: {ev.get('reason')}")

        except Exception:
            logger.exception("Evaluation/Langfuse logging failed")

    return answer, debug


# ----------------------------
# CLI
# ----------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the terminal QA application.

    This function defines and parses all CLI options required to configure
    the hybrid retrieval and question-answering pipeline, including retrieval
    parameters, reranking options, hybrid weighting, and debug output control.

    Args:
        argv (Optional[List[str]]): Optional list of command-line arguments.
            If None, arguments are read from sys.argv.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing
        question text, retrieval configuration values, and output flags.
    """
    parser = argparse.ArgumentParser(
        prog="main",
        description="Terminal QA: Hybrid RAG + Agentic Routing (FAISS + BM25 + YAML prompts)",
    )
    parser.add_argument("-q", "--question", required=True, help="Question to ask")

    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--use-mmr", default=str(DEFAULT_USE_MMR).lower(), choices=["true", "false"])
    parser.add_argument("--mmr-lambda", type=float, default=DEFAULT_MMR_LAMBDA)

    parser.add_argument("--use-rerank", default=str(DEFAULT_USE_RERANK).lower(), choices=["true", "false"])
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)

    parser.add_argument("--hybrid-alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--hybrid-beta", type=float, default=DEFAULT_BETA)

    parser.add_argument("--show-sources", default="true", choices=["true", "false"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the terminal-based QA application.

    This function initializes the environment, parses CLI arguments, builds
    the retrieval configuration, loads required components (vector store,
    language model, BM25 corpus), and executes a question-answering workflow
    using a hybrid retrieval and optional reranking strategy.

    Args:
        argv (Optional[List[str]]): Optional list of command-line arguments.
            If None, arguments are read directly from sys.argv.

    Returns:
        int: Exit status code.
            Returns 0 on successful execution, or 1 if an error occurs
            (e.g., missing environment variables or runtime failure).
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print('OPENAI_API_KEY missing in environment (.env)')
        return 1

    args = _parse_args(argv)

    cfg = RetrievalConfig(
        top_k=int(args.top_k),
        fetch_k=int(args.fetch_k),
        use_mmr=(str(args.use_mmr).lower() == "true"),
        mmr_lambda=float(args.mmr_lambda),
        use_rerank=(str(args.use_rerank).lower() == "true"),
        rerank_top_k=int(args.rerank_top_k),
        hybrid_alpha=float(args.hybrid_alpha),
        hybrid_beta=float(args.hybrid_beta),
    )

    try:
        vs = load_vectorstore(VECTORSTORE_PATH, EMBEDDING_MODEL)
        llm = init_llm(DEFAULT_LLM_MODEL)
        bm25_corpus = load_all_chunks_for_bm25(vs)

        answer, debug = answer_question(
            args.question,
            vs=vs,
            llm=llm,
            cfg=cfg,
            bm25_corpus=bm25_corpus,
        )

        print("\nANSWER:\n")
        print(answer)

        if str(args.show_sources).lower() == "true":
            print("\nDEBUG / SOURCES:\n")
            print(json.dumps(debug, ensure_ascii=False, indent=2))

        return 0
    except Exception as e:
        print("Query failed")
        print(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))