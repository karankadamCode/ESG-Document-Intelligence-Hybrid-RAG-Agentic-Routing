# tests/test_rag_smoke.py
import pytest
from pathlib import Path
import sys

# Add project root (parent of tests/) to sys.path so `import main` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main


@pytest.mark.smoke
def test_not_found_message_is_exact(vs_and_bm25, llm, default_cfg):
    vs, bm25 = vs_and_bm25

    # Use a query that should not exist in an ESG report, to force Not found behavior.
    ans, debug = main.answer_question(
        "What is the intergalactic trade license number mentioned in this report?",
        vs=vs,
        llm=llm,
        cfg=default_cfg,
        bm25_corpus=bm25,
    )

    assert ans == "Not found in provided documents."


@pytest.mark.smoke
def test_router_returns_valid_modes(llm):
    rd = main.route_query("Summarize climate risk management approach.", llm)
    assert rd.mode in {"rag", "summarize", "extract_kpi"}
    assert rd.retrieval_hint in {"hybrid", "vector", "lexical"}