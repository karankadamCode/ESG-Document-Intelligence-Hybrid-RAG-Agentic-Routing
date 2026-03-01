# tests/test_llm_evals.py
import os
import pytest
from pathlib import Path
import sys

# Add project root (parent of tests/) to sys.path so `import main` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main

pytestmark = pytest.mark.llm


def _can_run_llm_evals():
    return os.getenv("RUN_LLM_EVALS", "false").lower() in {"1", "true", "yes", "y"}


@pytest.mark.skipif(not _can_run_llm_evals(), reason="Set RUN_LLM_EVALS=true to run LLM evals")
def test_deepeval_faithfulness_turn(vs_and_bm25, llm, default_cfg):
    from evals import run_deepeval_turn

    vs, bm25 = vs_and_bm25

    ans, debug = main.answer_question(
        "Summarize climate risk management approach.",
        vs=vs,
        llm=llm,
        cfg=default_cfg,
        bm25_corpus=bm25,
    )

    retrieval_context = debug.get("retrieval_context", [])
    assert isinstance(retrieval_context, list)

    result = run_deepeval_turn(
        question="Summarize climate risk management approach.",
        answer=ans,
        retrieval_context=retrieval_context,
    )

    assert isinstance(result, dict)
    assert result.get("enabled") is True

    metrics = result.get("metrics", {}) or {}
    assert "faithfulness" in metrics

    score = float(metrics["faithfulness"].get("score", 0.0))
    assert 0.0 <= score <= 1.0