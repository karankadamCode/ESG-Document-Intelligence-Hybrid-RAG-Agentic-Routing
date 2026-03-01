from __future__ import annotations

import os
import time
import hashlib
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import main


app = FastAPI(title="ESG Document Intelligence System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

_VS: Optional[Any] = None
_BM25: Optional[List[Any]] = None
_LLM: Optional[Any] = None


class QADebug(BaseModel):
    mode: str
    retrieval_hint: str
    retrieval_result: Optional[str] = None
    cfg: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    retrieval_context: Optional[List[str]] = None
    timestamp_utc: str


class QAResponse(BaseModel):
    query: str
    answer: str
    debug: QADebug


_CACHE_ENABLED = os.getenv("API_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
_CACHE_TTL_SECONDS = int(os.getenv("API_CACHE_TTL_SECONDS", "300"))
_CACHE_MAX_ITEMS = int(os.getenv("API_CACHE_MAX_ITEMS", "256"))
_QA_CACHE: Dict[str, Dict[str, Any]] = {}


def _make_cache_key(q: str) -> str:
    cfg_fingerprint = "|".join(
        [
            str(main.DEFAULT_LLM_MODEL),
            str(main.EMBEDDING_MODEL),
            str(main.DEFAULT_TOP_K),
            str(main.DEFAULT_USE_MMR),
            str(main.DEFAULT_USE_RERANK),
            str(main.DEFAULT_ALPHA),
            str(main.DEFAULT_BETA),
        ]
    )
    raw = (q or "").strip().lower() + "|" + cfg_fingerprint
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not _CACHE_ENABLED:
        return None
    item = _QA_CACHE.get(key)
    if not item:
        return None
    if float(item.get("expires_at", 0.0)) < time.time():
        _QA_CACHE.pop(key, None)
        return None
    return item.get("payload")


def _cache_set(key: str, payload: Dict[str, Any]) -> None:
    if not _CACHE_ENABLED:
        return
    if len(_QA_CACHE) >= _CACHE_MAX_ITEMS:
        _QA_CACHE.pop(next(iter(_QA_CACHE)), None)
    _QA_CACHE[key] = {"expires_at": time.time() + _CACHE_TTL_SECONDS, "payload": payload}


def _init_once() -> Tuple[Any, List[Any], Any]:
    global _VS, _BM25, _LLM

    if _VS is not None and _BM25 is not None and _LLM is not None:
        return _VS, _BM25, _LLM

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError('OPENAI_API_KEY missing in environment (.env)')

    _VS = main.load_vectorstore(main.VECTORSTORE_PATH, main.EMBEDDING_MODEL)
    _BM25 = main.load_all_chunks_for_bm25(_VS)
    _LLM = main.init_llm(main.DEFAULT_LLM_MODEL)

    return _VS, _BM25, _LLM


@app.get("/health")
def health(response: Response) -> Dict[str, str]:
    response.headers["Cache-Control"] = "no-store"
    return {"status": "ok"}


def _run_evals_background(q_clean: str, answer: str, debug: Dict[str, Any]) -> None:
    """
    Runs evals without blocking HTTP response.

    Expectation (recommended):
    - main.run_evals(...) or similar exists and does not mutate the answer.
    - If it does not exist, this safely does nothing.
    """
    try:
        if hasattr(main, "run_evals"):
            main.run_evals(
                query=q_clean,
                answer=answer,
                debug=debug,
            )
    except Exception:
        # never crash request flow due to eval failures
        return


@app.get("/qa", response_model=QAResponse)
async def qa(
    response: Response,
    background_tasks: BackgroundTasks,
    q: str = Query(..., min_length=1, description="User query/question"),
) -> Dict[str, Any]:
    try:
        q_clean = (q or "").strip()
        if not q_clean:
            raise ValueError("Question is empty")

        cache_key = _make_cache_key(q_clean)
        cached = _cache_get(cache_key)
        if cached is not None:
            response.headers["Cache-Control"] = f"public, max-age={_CACHE_TTL_SECONDS}"
            response.headers["X-Cache"] = "HIT"
            return cached

        vs, bm25_corpus, llm = _init_once()
        cfg = main.RetrievalConfig()

        # Critical: disable evals inside answer path (only if your cfg supports it)
        # Add this flag in main.RetrievalConfig if it does not exist today.
        if hasattr(cfg, "enable_evals"):
            cfg.enable_evals = False
        if hasattr(cfg, "run_evals"):
            cfg.run_evals = False

        # Run blocking work off the event loop
        answer, debug = await asyncio.to_thread(
            main.answer_question,
            q_clean,
            vs=vs,
            llm=llm,
            cfg=cfg,
            bm25_corpus=bm25_corpus,
        )

        payload: Dict[str, Any] = {
            "query": q_clean,
            "answer": answer,
            "debug": debug,
        }

        _cache_set(cache_key, payload)

        # Run evals in parallel (does not block response)
        background_tasks.add_task(_run_evals_background, q_clean, answer, debug)

        response.headers["Cache-Control"] = f"public, max-age={_CACHE_TTL_SECONDS}"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Evals"] = "BACKGROUND"
        return payload

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Vectorstore missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))