import pytest
from pathlib import Path
import sys

# Add project root (parent of tests/) to sys.path so `import main` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import main


@pytest.mark.smoke
def test_answer_question_returns_debug_contract(vs_and_bm25, llm, default_cfg):
    vs, bm25 = vs_and_bm25

    ans, debug = main.answer_question(
        "Summarize climate risk management approach.",
        vs=vs,
        llm=llm,
        cfg=default_cfg,
        bm25_corpus=bm25,
    )

    assert isinstance(ans, str)
    assert isinstance(debug, dict)

    # Basic debug contract
    assert "mode" in debug
    assert "retrieval_hint" in debug
    assert "retrieval" in debug
    assert "timestamp_utc" in debug

    # Retrieval payload contract
    r = debug["retrieval"]
    assert "sources" in r and isinstance(r["sources"], list)
    assert "pages" in r and isinstance(r["pages"], list)
    assert "chunks_used" in r

    # Answer should not be empty
    assert ans.strip() != ""